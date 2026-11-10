// cortex — replacement policies, replay simulator, predictive prefetcher,
//          copy-on-write registry, admission control, write-back buffer.
#include "cortex/cortex.h"

#include <algorithm>
#include <cstdio>
#include <deque>
#include <list>
#include <unordered_set>

namespace cortex {

const char * evict_policy_name(evict_policy p) {
    switch (p) {
        case evict_policy::lru:     return "lru";
        case evict_policy::clock:   return "clock";
        case evict_policy::two_q:   return "2q";
        case evict_policy::s3fifo:  return "s3fifo";
        case evict_policy::refault: return "refault";
    }
    return "unknown";
}

namespace {

// ===========================================================================
// LRU — baseline. Recency only; every scan pollutes it.
// ===========================================================================
class lru_index final : public eviction_index {
public:
    explicit lru_index(size_t cap) : cap_(cap) {}

    const char * name() const override { return "lru"; }
    bool   contains(uint64_t k) const override { return map_.count(k) != 0; }
    size_t resident_bytes() const override { return bytes_; }
    size_t size() const override { return map_.size(); }
    size_t capacity() const override { return cap_; }
    void   set_capacity(size_t b) override { cap_ = b; }

    bool touch(uint64_t k) override {
        auto it = map_.find(k);
        if (it == map_.end()) { return false; }
        order_.splice(order_.begin(), order_, it->second.it);
        it->second.it = order_.begin();
        return true;
    }

    std::vector<uint64_t> evict_to_fit(size_t needed) override {
        std::vector<uint64_t> ev;
        while (bytes_ + needed > cap_ && !order_.empty()) {
            const uint64_t k = order_.back();
            auto it = map_.find(k);
            bytes_ -= it->second.bytes;
            order_.pop_back();
            map_.erase(it);
            ev.push_back(k);
        }
        return ev;
    }

    std::vector<uint64_t> insert(uint64_t k, size_t bytes) override {
        std::vector<uint64_t> ev;
        auto it = map_.find(k);
        if (it != map_.end()) {
            bytes_ -= it->second.bytes;
            it->second.bytes = bytes;
            bytes_ += bytes;
            order_.splice(order_.begin(), order_, it->second.it);
            it->second.it = order_.begin();
        } else if (bytes > cap_) {
            return evict_to_fit(cap_); // cannot ever fit
        } else {
            auto e = evict_to_fit(bytes);
            ev.insert(ev.end(), e.begin(), e.end());
            order_.push_front(k);
            map_[k] = ent{ order_.begin(), bytes };
            bytes_ += bytes;
        }
        auto e2 = evict_to_fit(0);
        ev.insert(ev.end(), e2.begin(), e2.end());
        return ev;
    }

private:
    struct ent { std::list<uint64_t>::iterator it; size_t bytes; };
    size_t cap_;
    size_t bytes_ = 0;
    std::list<uint64_t> order_; // front = MRU
    std::unordered_map<uint64_t, ent> map_;
};

// ===========================================================================
// CLOCK — second chance. Cheap approximation of LRU with a reference bit.
// ===========================================================================
class clock_index final : public eviction_index {
public:
    explicit clock_index(size_t cap) : cap_(cap) {}

    const char * name() const override { return "clock"; }
    bool   contains(uint64_t k) const override { return pos_.count(k) != 0; }
    size_t resident_bytes() const override { return bytes_; }
    size_t size() const override { return pos_.size(); }
    size_t capacity() const override { return cap_; }
    void   set_capacity(size_t b) override { cap_ = b; }

    bool touch(uint64_t k) override {
        auto it = pos_.find(k);
        if (it == pos_.end()) { return false; }
        slots_[it->second].ref = true; // give it a second chance
        return true;
    }

    std::vector<uint64_t> evict_to_fit(size_t needed) override {
        std::vector<uint64_t> ev;
        while (bytes_ + needed > cap_ && !pos_.empty()) {
            if (!evict_one(ev)) { break; }
        }
        return ev;
    }

    std::vector<uint64_t> insert(uint64_t k, size_t bytes) override {
        std::vector<uint64_t> ev;
        auto it = pos_.find(k);
        if (it != pos_.end()) {
            slot & s = slots_[it->second];
            bytes_ -= s.bytes;
            s.bytes = bytes;
            bytes_ += bytes;
            s.ref = true;
            auto e = evict_to_fit(0);
            ev.insert(ev.end(), e.begin(), e.end());
            return ev;
        }
        if (bytes > cap_) {
            return evict_to_fit(cap_);
        }
        auto e = evict_to_fit(bytes);
        ev.insert(ev.end(), e.begin(), e.end());

        size_t idx;
        if (!freelist_.empty()) { idx = freelist_.back(); freelist_.pop_back(); }
        else                    { slots_.push_back(slot{}); idx = slots_.size() - 1; }
        slots_[idx] = slot{ k, bytes, true, true };
        pos_[k]     = idx;
        bytes_     += bytes;
        return ev;
    }

private:
    struct slot { uint64_t key = 0; size_t bytes = 0; bool ref = false; bool used = false; };

    bool evict_one(std::vector<uint64_t> & ev) {
        if (slots_.empty()) { return false; }
        const size_t bound = slots_.size() * 2 + 8;
        for (size_t scanned = 0; scanned < bound; scanned++) {
            if (hand_ >= slots_.size()) { hand_ = 0; }
            slot & s = slots_[hand_];
            if (!s.used) { hand_++; continue; }
            if (s.ref)   { s.ref = false; hand_++; continue; }
            ev.push_back(s.key);
            bytes_ -= s.bytes;
            pos_.erase(s.key);
            s.used = false;
            freelist_.push_back(hand_);
            hand_++;
            return true;
        }
        return false;
    }

    size_t cap_;
    size_t bytes_ = 0;
    size_t hand_  = 0;
    std::vector<slot> slots_;
    std::vector<size_t> freelist_;
    std::unordered_map<uint64_t, size_t> pos_;
};

// ===========================================================================
// 2Q — A1in (probationary FIFO) / A1out (ghost) / Am (protected LRU).
// The ghost queue is what keeps a one-shot scan from evicting the hot set.
// ===========================================================================
class twoq_index final : public eviction_index {
public:
    explicit twoq_index(size_t cap) : cap_(cap) { set_capacity(cap); }

    const char * name() const override { return "2q"; }
    bool   contains(uint64_t k) const override {
        return a1in_sz_.count(k) != 0 || am_.count(k) != 0;
    }
    size_t resident_bytes() const override { return bytes_; }
    size_t size() const override { return a1in_sz_.size() + am_.size(); }
    size_t capacity() const override { return cap_; }
    void   set_capacity(size_t b) override { cap_ = b; kin_ = b / 4; }

    bool touch(uint64_t k) override {
        auto ia = a1in_sz_.find(k);
        if (ia != a1in_sz_.end()) { return true; } // FIFO: no reordering
        auto im = am_.find(k);
        if (im != am_.end()) {
            am_order_.splice(am_order_.begin(), am_order_, im->second.it);
            im->second.it = am_order_.begin();
            return true;
        }
        return false;
    }

    std::vector<uint64_t> insert(uint64_t k, size_t bytes) override {
        std::vector<uint64_t> ev;
        if (contains(k)) { touch(k); return ev; }

        if (ghost_set_.count(k)) { // seen twice within a lifetime -> protected
            ghost_set_.erase(k);
            insert_am(k, bytes, ev);
        } else {
            insert_a1in(k, bytes, ev);
        }
        auto e = evict_to_fit(0);
        ev.insert(ev.end(), e.begin(), e.end());
        return ev;
    }

    std::vector<uint64_t> evict_to_fit(size_t needed) override {
        std::vector<uint64_t> ev;
        size_t guard = 0;
        while (bytes_ + needed > cap_ && (a1in_sz_.size() + am_.size()) > 0 && guard++ < (1u << 20)) {
            if (!a1in_.empty()) {                 // evict probationary first
                evict_a1in_head(ev);
            } else if (!am_order_.empty()) {      // then LRU of the protected set
                const uint64_t k = am_order_.back();
                auto it = am_.find(k);
                bytes_ -= it->second.bytes;
                am_order_.pop_back();
                am_.erase(it);
                ev.push_back(k);
            } else {
                break;
            }
        }
        return ev;
    }

private:
    struct ent { std::list<uint64_t>::iterator it; size_t bytes; };

    void insert_am(uint64_t k, size_t bytes, std::vector<uint64_t> & ev) {
        auto e = evict_to_fit(bytes);
        ev.insert(ev.end(), e.begin(), e.end());
        am_order_.push_front(k);
        am_[k] = ent{ am_order_.begin(), bytes };
        bytes_ += bytes;
    }

    void insert_a1in(uint64_t k, size_t bytes, std::vector<uint64_t> & ev) {
        a1in_.push_back(k);
        a1in_sz_[k]   = bytes;
        a1in_bytes_  += bytes;
        bytes_       += bytes;

        while (a1in_bytes_ > kin_ && !a1in_.empty()) {
            const uint64_t g = a1in_.front();
            a1in_.pop_front();
            const size_t gs = a1in_sz_[g];
            a1in_sz_.erase(g);
            a1in_bytes_ -= gs;
            bytes_      -= gs;
            ghost_q_.push_back(g);
            ghost_set_.insert(g);
            while (ghost_q_.size() > max_ghost_) {
                ghost_set_.erase(ghost_q_.front());
                ghost_q_.pop_front();
            }
            ev.push_back(g); // demoted out of the resident set
        }
    }

    void evict_a1in_head(std::vector<uint64_t> & ev) {
        const uint64_t k = a1in_.front();
        a1in_.pop_front();
        const size_t s = a1in_sz_[k];
        a1in_sz_.erase(k);
        a1in_bytes_ -= s;
        bytes_      -= s;
        ev.push_back(k);
    }

    size_t cap_;
    size_t kin_        = 0;
    size_t bytes_      = 0;
    size_t a1in_bytes_ = 0;
    size_t max_ghost_  = 4096;

    std::deque<uint64_t>  a1in_;
    std::unordered_map<uint64_t, size_t> a1in_sz_;

    std::deque<uint64_t>  ghost_q_;
    std::unordered_set<uint64_t> ghost_set_;

    std::list<uint64_t> am_order_;
    std::unordered_map<uint64_t, ent> am_;
};

// ===========================================================================
// S3-FIFO — small FIFO (S) + main FIFO (M) + ghost (G) with lazy promotion.
// Beats LRU on scan-heavy / skewed workloads because it never moves hot
// entries on access; promotion is deferred until eviction time.
// ===========================================================================
class s3fifo_index final : public eviction_index {
public:
    explicit s3fifo_index(size_t cap) : cap_(cap) { set_capacity(cap); }

    const char * name() const override { return "s3fifo"; }
    bool   contains(uint64_t k) const override { return sz_.count(k) != 0; }
    size_t resident_bytes() const override { return bytes_; }
    size_t size() const override { return sz_.size(); }
    size_t capacity() const override { return cap_; }
    void   set_capacity(size_t b) override { cap_ = b; s_cap_ = b / 10; }

    bool touch(uint64_t k) override {
        auto it = freq_.find(k);
        if (it == freq_.end()) { return false; }
        it->second = 1; // set the reference bit; promotion happens lazily on eviction
        return true;
    }

    std::vector<uint64_t> insert(uint64_t k, size_t bytes) override {
        std::vector<uint64_t> ev;
        if (sz_.count(k)) { touch(k); return ev; }

        if (g_set_.count(k)) { // ghost hit: it was evicted from S recently
            g_set_.erase(k);
            m_q_.push_back(k);
            freq_[k] = 0;
            sz_[k]   = bytes;
            bytes_  += bytes;
        } else {
            s_q_.push_back(k);
            freq_[k]   = 0;
            sz_[k]     = bytes;
            bytes_    += bytes;
            s_bytes_  += bytes;
        }

        while (s_bytes_ > s_cap_ && s_q_.size() > 1) { evict_from_s(ev); }
        auto e = evict_to_fit(0);
        ev.insert(ev.end(), e.begin(), e.end());
        return ev;
    }

    std::vector<uint64_t> evict_to_fit(size_t needed) override {
        std::vector<uint64_t> ev;
        size_t guard = 0;
        while (bytes_ + needed > cap_ && !sz_.empty() && guard++ < (1u << 20)) {
            if (!m_q_.empty())      { evict_from_m(ev); }
            else if (!s_q_.empty()) { evict_from_s(ev); }
            else                    { break; }
        }
        return ev;
    }

private:
    void drop(uint64_t k, std::vector<uint64_t> & ev, bool ghost) {
        const size_t s = sz_[k];
        sz_.erase(k);
        freq_.erase(k);
        bytes_ -= s;
        ev.push_back(k);
        if (ghost) {
            g_q_.push_back(k);
            g_set_.insert(k);
            while (g_q_.size() > max_ghost_) {
                g_set_.erase(g_q_.front());
                g_q_.pop_front();
            }
        }
    }

    void evict_from_s(std::vector<uint64_t> & ev) {
        if (s_q_.empty()) { return; }
        const uint64_t k = s_q_.front();
        s_q_.pop_front();
        const size_t s = sz_[k];
        s_bytes_ -= s;
        if (freq_[k]) {           // reused -> promote to main
            freq_[k] = 0;
            m_q_.push_back(k);
            return;
        }
        drop(k, ev, /*ghost=*/true);
    }

    void evict_from_m(std::vector<uint64_t> & ev) {
        while (!m_q_.empty()) {
            const uint64_t k = m_q_.front();
            m_q_.pop_front();
            if (freq_[k]) {           // reused -> second chance, back of the line
                freq_[k] = 0;
                m_q_.push_back(k);
                continue;
            }
            drop(k, ev, /*ghost=*/false);
            return;
        }
    }

    size_t cap_;
    size_t s_cap_     = 0;
    size_t max_ghost_ = 4096;
    size_t bytes_     = 0;
    size_t s_bytes_   = 0;

    std::deque<uint64_t> s_q_;
    std::deque<uint64_t> m_q_;
    std::deque<uint64_t> g_q_;
    std::unordered_set<uint64_t> g_set_;
    std::unordered_map<uint64_t, size_t>  sz_;
    std::unordered_map<uint64_t, uint8_t> freq_;
};

// ===========================================================================
// refault — MGLRU-inspired. Two generations (unprotected / protected) plus a
// reuse-distance window: a block that is evicted and then demanded again
// *quickly* is thrashing, so it is promoted straight into the protected
// generation and is safe from the next sweep of cold blocks. This is the
// anti-thrash property plain LRU lacks.
// ===========================================================================
class refault_index final : public eviction_index {
public:
    explicit refault_index(size_t cap) : cap_(cap) {}

    const char * name() const override { return "refault"; }
    bool   contains(uint64_t k) const override { return map_.count(k) != 0; }
    size_t resident_bytes() const override { return bytes_; }
    size_t size() const override { return map_.size(); }
    size_t capacity() const override { return cap_; }
    void   set_capacity(size_t b) override { cap_ = b; }

    bool touch(uint64_t k) override {
        ++tick_;
        auto it = map_.find(k);
        if (it == map_.end()) { return false; }
        promote(it->second); // an access promotes into the protected generation
        return true;
    }

    std::vector<uint64_t> insert(uint64_t k, size_t bytes) override {
        ++tick_;
        std::vector<uint64_t> ev;
        auto it = map_.find(k);
        if (it != map_.end()) { promote(it->second); return ev; }

        // refault detection: was this key evicted recently?
        bool refaulting = false;
        auto ri = refault_.find(k);
        if (ri != refault_.end()) {
            refaulting = (tick_ - ri->second) <= window_;
            refault_.erase(k);
        }

        auto e = evict_to_fit(bytes);
        ev.insert(ev.end(), e.begin(), e.end());

        if (refaulting) {
            prot_order_.push_front(k);
            map_[k] = ent{ prot_order_.begin(), true, bytes };
        } else {
            unprot_order_.push_front(k);
            map_[k] = ent{ unprot_order_.begin(), false, bytes };
        }
        bytes_ += bytes;

        auto e2 = evict_to_fit(0);
        ev.insert(ev.end(), e2.begin(), e2.end());
        return ev;
    }

    std::vector<uint64_t> evict_to_fit(size_t needed) override {
        std::vector<uint64_t> ev;
        size_t guard = 0;
        while (bytes_ + needed > cap_ && !map_.empty() && guard++ < (1u << 20)) {
            if (!unprot_order_.empty()) {
                evict_one(unprot_order_, unprot_order_.back(), ev);
            } else if (!prot_order_.empty()) {
                evict_one(prot_order_, prot_order_.back(), ev);
            } else {
                break;
            }
        }
        return ev;
    }

private:
    struct ent { std::list<uint64_t>::iterator it; bool prot; size_t bytes; };

    void promote(ent & e) {
        if (e.prot) {
            prot_order_.splice(prot_order_.begin(), prot_order_, e.it);
            e.it = prot_order_.begin();
            return;
        }
        const uint64_t k = *e.it;
        unprot_order_.erase(e.it);
        prot_order_.push_front(k);
        e.it   = prot_order_.begin();
        e.prot = true;
    }

    void evict_one(std::list<uint64_t> & lst, uint64_t k, std::vector<uint64_t> & ev) {
        lst.pop_back();
        auto it = map_.find(k);
        if (it == map_.end()) { return; }
        bytes_ -= it->second.bytes;
        map_.erase(it);
        ev.push_back(k);

        refault_[k] = tick_;
        refault_q_.push_back(k);
        while (refault_q_.size() > max_refault_) {
            const uint64_t old = refault_q_.front();
            refault_q_.pop_front();
            refault_.erase(old);
        }
    }

    size_t cap_;
    size_t bytes_       = 0;
    uint64_t tick_      = 0;
    uint64_t window_    = 4096;
    size_t max_refault_ = 8192;

    std::list<uint64_t> unprot_order_;
    std::list<uint64_t> prot_order_;
    std::unordered_map<uint64_t, ent> map_;

    std::unordered_map<uint64_t, uint64_t> refault_;
    std::deque<uint64_t> refault_q_;
};

} // namespace

std::unique_ptr<eviction_index> make_eviction(evict_policy p, size_t capacity_bytes) {
    switch (p) {
        case evict_policy::lru:     return std::make_unique<lru_index>(capacity_bytes);
        case evict_policy::clock:   return std::make_unique<clock_index>(capacity_bytes);
        case evict_policy::two_q:   return std::make_unique<twoq_index>(capacity_bytes);
        case evict_policy::s3fifo:  return std::make_unique<s3fifo_index>(capacity_bytes);
        case evict_policy::refault: return std::make_unique<refault_index>(capacity_bytes);
    }
    return std::make_unique<lru_index>(capacity_bytes);
}

// ===========================================================================
// trace replay
// ===========================================================================

sim_result simulate(eviction_index & idx,
                    const std::vector<sim_access> & trace,
                    const memory_hierarchy & h,
                    tier from) {
    sim_result r;
    for (const auto & a : trace) {
        r.accesses++;
        if (idx.touch(a.key)) {
            r.hits++;
        } else {
            r.misses++;
            r.bytes_moved += double(a.bytes);
            r.est_time_us += h.transfer_cost_us(from, tier::vram, a.bytes);
            auto ev = idx.insert(a.key, a.bytes);
            r.evictions += ev.size();
        }
        r.peak_resident = std::max(r.peak_resident, idx.resident_bytes());
    }
    r.hit_rate = r.accesses > 0 ? double(r.hits) / double(r.accesses) : 0.0;
    return r;
}

// ===========================================================================
// predictive readahead
// ===========================================================================

void expert_prefetcher::observe(uint32_t layer, const std::vector<uint32_t> & experts) {
    n_obs_++;
    if (layer > 0) {
        auto prev = last_.find(layer - 1);
        if (prev != last_.end()) {
            for (uint32_t from : prev->second) {
                auto & per_to = trans_[layer - 1][from];
                for (uint32_t to : experts) {
                    per_to[to]++;
                }
                from_total_[layer - 1][from]++;
            }
        }
    }
    last_[layer] = experts;
}

double expert_prefetcher::confidence(uint32_t layer, uint32_t expert) const {
    if (layer == 0) { return 0.0; }
    auto prev = last_.find(layer - 1);
    if (prev == last_.end()) { return 0.0; }
    auto tl = trans_.find(layer - 1);
    if (tl == trans_.end()) { return 0.0; }
    auto ft = from_total_.find(layer - 1);
    if (ft == from_total_.end()) { return 0.0; }

    double best = 0.0;
    for (uint32_t from : prev->second) {
        auto tf = tl->second.find(from);
        if (tf == tl->second.end()) { continue; }
        auto tot = ft->second.find(from);
        if (tot == ft->second.end() || tot->second == 0) { continue; }
        auto ce = tf->second.find(expert);
        if (ce == tf->second.end()) { continue; }
        best = std::max(best, double(ce->second) / double(tot->second));
    }
    return best;
}

std::vector<uint32_t> expert_prefetcher::predict(uint32_t layer, double min_confidence) const {
    std::vector<uint32_t> out;
    if (layer == 0) { return out; }
    auto prev = last_.find(layer - 1);
    if (prev == last_.end()) { return out; }
    auto tl = trans_.find(layer - 1);
    if (tl == trans_.end()) { return out; }
    auto ft = from_total_.find(layer - 1);
    if (ft == from_total_.end()) { return out; }

    std::unordered_map<uint32_t, double> score;
    for (uint32_t from : prev->second) {
        auto tf = tl->second.find(from);
        if (tf == tl->second.end()) { continue; }
        auto tot = ft->second.find(from);
        if (tot == ft->second.end() || tot->second == 0) { continue; }
        for (const auto & kv : tf->second) {
            const double p = double(kv.second) / double(tot->second);
            auto it = score.find(kv.first);
            if (it == score.end()) { score[kv.first] = p; }
            else                   { it->second = std::max(it->second, p); }
        }
    }

    for (const auto & kv : score) {
        if (kv.second >= min_confidence) { out.push_back(kv.first); }
    }
    std::sort(out.begin(), out.end(),
              [&](uint32_t a, uint32_t b) { return score.at(a) > score.at(b); });
    return out;
}

// ===========================================================================
// copy-on-write registry
// ===========================================================================

uint64_t cow_registry::acquire(uint64_t content_hash, size_t bytes) {
    auto hit = by_hash_.find(content_hash);
    if (hit != by_hash_.end()) {              // shared with an existing block
        blocks_[hit->second].refcount++;
        logical_bytes_ += bytes;
        return hit->second;
    }
    const uint64_t handle = next_handle_++;
    blocks_[handle] = block{ content_hash, bytes, 1 };
    by_hash_[content_hash] = handle;
    physical_bytes_ += bytes;
    logical_bytes_  += bytes;
    return handle;
}

bool cow_registry::release(uint64_t handle) {
    auto it = blocks_.find(handle);
    if (it == blocks_.end()) { return false; }
    logical_bytes_ -= it->second.bytes;
    if (it->second.refcount > 0) { it->second.refcount--; }
    if (it->second.refcount == 0) {
        physical_bytes_ -= it->second.bytes;
        by_hash_.erase(it->second.hash);
        blocks_.erase(it);
        return true;
    }
    return false;
}

size_t cow_registry::refcount(uint64_t handle) const {
    auto it = blocks_.find(handle);
    return it == blocks_.end() ? 0 : it->second.refcount;
}

size_t cow_registry::physical_bytes() const { return physical_bytes_; }
size_t cow_registry::logical_bytes()  const { return logical_bytes_; }

double cow_registry::dedup_ratio() const {
    return physical_bytes_ > 0 ? double(logical_bytes_) / double(physical_bytes_) : 1.0;
}

std::vector<uint64_t> cow_registry::evictable() const {
    std::vector<uint64_t> out;
    for (const auto & kv : blocks_) {
        if (kv.second.refcount == 0) { out.push_back(kv.first); }
    }
    return out;
}

// ===========================================================================
// admission control
// ===========================================================================

void admission_controller::on_hit(uint64_t key) {
    double & f = freq_[key];
    f = std::min(f * 2.0 + 1.0, 1e9);
}

void admission_controller::on_step(double pressure) {
    pressure_ = std::max(0.0, std::min(1.0, pressure));
    for (auto it = freq_.begin(); it != freq_.end();) {
        it->second *= cfg_.decay_per_step;
        if (it->second < 1e-3) { it = freq_.erase(it); }
        else                   { ++it; }
    }
}

double admission_controller::effective_threshold() const {
    if (pressure_ <= cfg_.pressure_high) { return cfg_.freq_threshold; }
    const double span = 1.0 - cfg_.pressure_high;
    const double t    = span > 0.0 ? (pressure_ - cfg_.pressure_high) / span : 1.0;
    return cfg_.freq_threshold * (1.0 + (cfg_.pressure_bar_scale - 1.0) * t);
}

bool admission_controller::should_promote(uint64_t key) const {
    auto it = freq_.find(key);
    if (it == freq_.end()) { return false; }
    return it->second >= effective_threshold();
}

double admission_controller::frequency(uint64_t key) const {
    auto it = freq_.find(key);
    return it == freq_.end() ? 0.0 : it->second;
}

// ===========================================================================
// write-back buffer
// ===========================================================================

void writeback_buffer::mark_dirty(uint64_t block, size_t bytes) {
    auto it = sizes_.find(block);
    if (it != sizes_.end()) {
        dirty_bytes_ -= it->second;
        it->second = bytes;
        dirty_bytes_ += bytes;
        return;
    }
    sizes_[block] = bytes;
    order_.push_back(block);
    dirty_bytes_ += bytes;
}

bool writeback_buffer::should_flush() const {
    return dirty_bytes_ >= max_dirty_bytes_ || order_.size() >= max_blocks_;
}

std::vector<uint64_t> writeback_buffer::flush() {
    std::vector<uint64_t> out;
    out.swap(order_);
    sizes_.clear();
    dirty_bytes_ = 0;
    flushes_++;
    return out;
}

} // namespace cortex
