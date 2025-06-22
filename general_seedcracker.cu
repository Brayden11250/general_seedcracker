#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstdint>
#include <map>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <memory>
#include <numeric>

#include <cuda_runtime.h>

// =======================================================================
// 1. CUDA Error Checking & Utility
// =======================================================================

#define CUDA_CHECK(err) { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        cudaDeviceReset(); \
        exit(EXIT_FAILURE); \
    } \
}

// RAII wrapper for CUDA memory to ensure cudaFree is always called.
template<typename T>
class CudaBuffer {
private:
    T* d_ptr = nullptr;
    size_t count = 0;

public:
    CudaBuffer() = default;

    explicit CudaBuffer(size_t n) {
        if (n > 0) {
            CUDA_CHECK(cudaMalloc(&d_ptr, n * sizeof(T)));
            count = n;
        }
    }

    ~CudaBuffer() {
        if (d_ptr) {
            cudaFree(d_ptr);
        }
    }

    // Disable copy semantics
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    // Enable move semantics
    CudaBuffer(CudaBuffer&& other) noexcept : d_ptr(other.d_ptr), count(other.count) {
        other.d_ptr = nullptr;
        other.count = 0;
    }

    CudaBuffer& operator=(CudaBuffer&& other) noexcept {
        if (this != &other) {
            if (d_ptr) cudaFree(d_ptr);
            d_ptr = other.d_ptr;
            count = other.count;
            other.d_ptr = nullptr;
            other.count = 0;
        }
        return *this;
    }

    T* get() { return d_ptr; }
    const T* get() const { return d_ptr; }
    size_t size() const { return count; }

    void copy_to_device(const T* h_data, size_t n) {
        if (n > count) throw std::runtime_error("Copy size exceeds buffer capacity.");
        CUDA_CHECK(cudaMemcpy(d_ptr, h_data, n * sizeof(T), cudaMemcpyHostToDevice));
    }

    void copy_to_host(T* h_data, size_t n) const {
        if (n > count) throw std::runtime_error("Copy size exceeds buffer capacity.");
        CUDA_CHECK(cudaMemcpy(h_data, d_ptr, n * sizeof(T), cudaMemcpyDeviceToHost));
    }
    
    void memset(int value) {
        if (d_ptr) {
            CUDA_CHECK(cudaMemset(d_ptr, value, count * sizeof(T)));
        }
    }
};


__device__ inline int32_t floorDiv(int32_t a, int32_t n) {
    int32_t r = a / n;
    if ((a % n != 0) && ((a < 0) != (n < 0))) {
        r--;
    }
    return r;
}

// =======================================================================
// 2. Core Data Structures & Enums (Host and Device)
// =======================================================================

// --- Generic Enums ---
enum class BlockRotation {
    NONE = 0, CLOCKWISE_90 = 1, CLOCKWISE_180 = 2, COUNTERCLOCKWISE_90 = 3
};

// --- Ruined Portal Specific Enums ---
enum class BlockMirror { NONE, FRONT_BACK };
enum class BiomeCategory { MOUNTAINS = 1, DESERT = 2, JUNGLE = 3 };
enum class PortalType {
    PORTAL_1, PORTAL_2, PORTAL_3, PORTAL_4, PORTAL_5, PORTAL_6, PORTAL_7, PORTAL_8, PORTAL_9, PORTAL_10,
    GIANT_PORTAL_1, GIANT_PORTAL_2, GIANT_PORTAL_3
};

// --- Shipwreck Specific Enums ---
enum class ShipwreckType {
    INVALID = -1,
    RIGHTSIDEUP_BACKHALF, RIGHTSIDEUP_BACKHALF_DEGRADED, RIGHTSIDEUP_FRONTHALF, RIGHTSIDEUP_FRONTHALF_DEGRADED,
    RIGHTSIDEUP_FULL, RIGHTSIDEUP_FULL_DEGRADED, SIDEWAYS_BACKHALF, SIDEWAYS_BACKHALF_DEGRADED,
    SIDEWAYS_FRONTHALF, SIDEWAYS_FRONTHALF_DEGRADED, SIDEWAYS_FULL, SIDEWAYS_FULL_DEGRADED,
    UPSIDEDOWN_BACKHALF, UPSIDEDOWN_BACKHALF_DEGRADED, UPSIDEDOWN_FRONTHALF, UPSIDEDOWN_FRONTHALF_DEGRADED,
    UPSIDEDOWN_FULL, UPSIDEDOWN_FULL_DEGRADED, WITH_MAST, WITH_MAST_DEGRADED,
};

// --- Village Specific Enums (NEW) ---
enum class VillageType { PLAINS, DESERT, SAVANNA, TAIGA, SNOWY };
enum class VillageStartPiece {
    PLAINS_FOUNTAIN_01, PLAINS_MEETING_POINT_1, PLAINS_MEETING_POINT_2, PLAINS_MEETING_POINT_3,
    DESERT_MEETING_POINT_1, DESERT_MEETING_POINT_2, DESERT_MEETING_POINT_3,
    SAVANNA_MEETING_POINT_1, SAVANNA_MEETING_POINT_2, SAVANNA_MEETING_POINT_3, SAVANNA_MEETING_POINT_4,
    TAIGA_MEETING_POINT_1, TAIGA_MEETING_POINT_2,
    SNOWY_MEETING_POINT_1, SNOWY_MEETING_POINT_2, SNOWY_MEETING_POINT_3,
    UNKNOWN_PIECE
};

// --- Unified Constraint Structures ---
// MODIFIED: Added VILLAGE
enum class ConstraintType { SHIPWRECK, RUINED_PORTAL, VILLAGE };

struct RuinedPortalConstraintData {
    BlockRotation rotation;
    BlockMirror mirror;
    PortalType type;
    BiomeCategory category;
};

struct ShipwreckConstraintData {
    BlockRotation rotation;
    ShipwreckType type;
    bool isBeached;
};

// NEW: Village constraint data
struct VillageConstraintData {
    BlockRotation rotation;
    VillageStartPiece piece;
    VillageType type;
    bool is_abandoned;
};

// MODIFIED: Added VillageConstraintData to the union
struct Constraint {
    ConstraintType type;
    int32_t chunkX;
    int32_t chunkZ;
    union {
        ShipwreckConstraintData shipwreck;
        RuinedPortalConstraintData portal;
        VillageConstraintData village;
    };
};

// =======================================================================
// 3. Minecraft LCG & Constants
// =======================================================================

__constant__ int64_t LCG_MULT = 25214903917LL;
__constant__ int64_t LCG_ADD = 11LL;
__constant__ int64_t XOR_MASK = 25214903917LL;
__constant__ int64_t MASK_48 = (1LL << 48) - 1;
__constant__ int64_t LCG_MULT_INV = 246154705703781LL;
__constant__ int64_t MULT_A = 341873128712LL;
__constant__ int64_t MULT_B = 132897987541LL;
__device__ const int64_t PILLAR_MULT = 1540035429LL;
__device__ const int64_t PILLAR_ADD = 239479465LL;

// Shipwreck Constants
__constant__ int32_t SHIPWRECK_SPACING = 24;
__constant__ int32_t SHIPWRECK_SEPARATION = 4;
__constant__ int64_t SHIPWRECK_SALT = 165745295;
__constant__ int32_t OCEAN_TYPE_COUNT = 20;
__constant__ int32_t BEACHED_TYPE_COUNT = 11;
__constant__ ShipwreckType d_STRUCTURE_LOCATION_OCEAN[20];
__constant__ ShipwreckType d_STRUCTURE_LOCATION_BEACHED[11];

// Ruined Portal Constants
__constant__ int32_t PORTAL_SPACING = 40;
__constant__ int32_t PORTAL_SEPARATION = 15;
__constant__ int64_t RUINED_PORTAL_SALT = 34222645LL;

// Village Constants (NEW)
__constant__ int32_t VILLAGE_SPACING = 34;
__constant__ int32_t VILLAGE_SEPARATION = 8;
__constant__ int64_t VILLAGE_SALT = 10387312LL;

// =======================================================================
// 4. Device-Side Logic
// =======================================================================

// --- Standalone LCG for GPU ---
struct StandaloneChunkRand {
private:
    int64_t seed;
public:
    __device__ void setSeed(int64_t s) {
        seed = (s ^ XOR_MASK) & MASK_48;
    }
    __device__ int32_t next(int32_t bits) {
        seed = (seed * LCG_MULT + LCG_ADD) & MASK_48;
        return (int32_t)((uint64_t)seed >> (48 - bits));
    }
    __device__ int32_t nextInt(int32_t bound) {
        if (bound <= 0) return 0;
        if ((bound & -bound) == bound) return (int32_t)((bound * (int64_t)next(31)) >> 31);
        int32_t bits, val;
        do { bits = next(31); val = bits % bound; } while (bits - val + (bound - 1) < 0);
        return val;
    }
    __device__ int64_t nextLong() {
        return ((int64_t)next(32) << 32) + next(32);
    }
    __device__ float nextFloat() {
        return next(24) / 16777216.0f; // (1 << 24)
    }
    __device__ void setRegionSeed(int64_t structureSeed, int32_t regionX, int32_t regionZ, int64_t salt) {
        int64_t s = (long long)regionX * MULT_A + (long long)regionZ * MULT_B + structureSeed + salt;
        setSeed(s);
    }
    __device__ void setCarverSeed(int64_t worldSeed, int32_t chunkX, int32_t chunkZ) {
        setSeed(worldSeed);
        long long a = nextLong();
        long long b = nextLong();
        setSeed((long long)chunkX * a ^ (long long)chunkZ * b ^ worldSeed);
    }
};

// --- Validation Functions (called by kernels) ---

// NEW: Village property calculation, ported and adapted.
__device__ void get_village_properties_device(
    VillageStartPiece& out_piece, BlockRotation& out_rot, bool& out_abandoned,
    VillageType type, StandaloneChunkRand& rand
) {
    // Rotation is determined by the first call to nextInt(4) which is next(2)
    out_rot = (BlockRotation)rand.next(2);

    int t;
    out_piece = VillageStartPiece::UNKNOWN_PIECE;
    out_abandoned = false;

    switch (type) {
        case VillageType::PLAINS:
            t = rand.nextInt(204);
            if      (t <  50) { out_piece = VillageStartPiece::PLAINS_FOUNTAIN_01;     }
            else if (t < 100) { out_piece = VillageStartPiece::PLAINS_MEETING_POINT_1; }
            else if (t < 150) { out_piece = VillageStartPiece::PLAINS_MEETING_POINT_2; }
            else if (t < 200) { out_piece = VillageStartPiece::PLAINS_MEETING_POINT_3; }
            else { // Abandoned (2% chance)
                out_abandoned = true;
                if      (t < 201) { out_piece = VillageStartPiece::PLAINS_FOUNTAIN_01; }
                else if (t < 202) { out_piece = VillageStartPiece::PLAINS_MEETING_POINT_1; }
                else if (t < 203) { out_piece = VillageStartPiece::PLAINS_MEETING_POINT_2; }
                else              { out_piece = VillageStartPiece::PLAINS_MEETING_POINT_3; }
            }
            break;
        case VillageType::DESERT:
            t = rand.nextInt(250);
            if      (t <  98) { out_piece = VillageStartPiece::DESERT_MEETING_POINT_1; }
            else if (t < 196) { out_piece = VillageStartPiece::DESERT_MEETING_POINT_2; }
            else if (t < 245) { out_piece = VillageStartPiece::DESERT_MEETING_POINT_3; }
            else { // Abandoned (2% chance)
                out_abandoned = true;
                if      (t < 247) { out_piece = VillageStartPiece::DESERT_MEETING_POINT_1; }
                else if (t < 249) { out_piece = VillageStartPiece::DESERT_MEETING_POINT_2; }
                else              { out_piece = VillageStartPiece::DESERT_MEETING_POINT_3; }
            }
            break;
        case VillageType::SAVANNA:
            t = rand.nextInt(459);
            if      (t < 100) { out_piece = VillageStartPiece::SAVANNA_MEETING_POINT_1; }
            else if (t < 150) { out_piece = VillageStartPiece::SAVANNA_MEETING_POINT_2; }
            else if (t < 300) { out_piece = VillageStartPiece::SAVANNA_MEETING_POINT_3; }
            else if (t < 450) { out_piece = VillageStartPiece::SAVANNA_MEETING_POINT_4; }
            else { // Abandoned (2% chance)
                out_abandoned = true;
                if      (t < 452) { out_piece = VillageStartPiece::SAVANNA_MEETING_POINT_1; }
                else if (t < 453) { out_piece = VillageStartPiece::SAVANNA_MEETING_POINT_2; }
                else if (t < 456) { out_piece = VillageStartPiece::SAVANNA_MEETING_POINT_3; }
                else              { out_piece = VillageStartPiece::SAVANNA_MEETING_POINT_4; }
            }
            break;
        case VillageType::TAIGA:
            t = rand.nextInt(100);
            if      (t <  49) { out_piece = VillageStartPiece::TAIGA_MEETING_POINT_1; }
            else if (t <  98) { out_piece = VillageStartPiece::TAIGA_MEETING_POINT_2; }
            else { // Abandoned (2% chance)
                out_abandoned = true;
                if (t < 99) { out_piece = VillageStartPiece::TAIGA_MEETING_POINT_1; }
                else        { out_piece = VillageStartPiece::TAIGA_MEETING_POINT_2; }
            }
            break;
        case VillageType::SNOWY:
            t = rand.nextInt(306);
            if      (t < 100) { out_piece = VillageStartPiece::SNOWY_MEETING_POINT_1; }
            else if (t < 150) { out_piece = VillageStartPiece::SNOWY_MEETING_POINT_2; }
            else if (t < 300) { out_piece = VillageStartPiece::SNOWY_MEETING_POINT_3; }
            else { // Abandoned (2% chance)
                out_abandoned = true;
                if      (t < 302) { out_piece = VillageStartPiece::SNOWY_MEETING_POINT_1; }
                else if (t < 303) { out_piece = VillageStartPiece::SNOWY_MEETING_POINT_2; }
                else              { out_piece = VillageStartPiece::SNOWY_MEETING_POINT_3; }
            }
            break;
    }
}

// NEW: Full validation function for a village constraint
__device__ bool check_village_full(int64_t structureSeed, const Constraint& constraint, StandaloneChunkRand& rand) {
    // 1. Check if the chunk coordinates are correct for the region
    int32_t regX = floorDiv(constraint.chunkX, VILLAGE_SPACING);
    int32_t regZ = floorDiv(constraint.chunkZ, VILLAGE_SPACING);
    rand.setRegionSeed(structureSeed, regX, regZ, VILLAGE_SALT);
    
    int32_t offset = VILLAGE_SPACING - VILLAGE_SEPARATION;
    if (regX * VILLAGE_SPACING + rand.nextInt(offset) != constraint.chunkX) return false;
    if (regZ * VILLAGE_SPACING + rand.nextInt(offset) != constraint.chunkZ) return false;

    // 2. Check the properties based on the chunk-specific seed
    rand.setCarverSeed(structureSeed, constraint.chunkX, constraint.chunkZ);

    VillageStartPiece found_piece;
    BlockRotation found_rot;
    bool found_abandoned;

    get_village_properties_device(found_piece, found_rot, found_abandoned, constraint.village.type, rand);

    return found_piece == constraint.village.piece &&
           found_rot == constraint.village.rotation &&
           found_abandoned == constraint.village.is_abandoned;
}

__device__ bool check_shipwreck_full(int64_t structureSeed, const Constraint& constraint, StandaloneChunkRand& rand) {
    int32_t regX = floorDiv(constraint.chunkX, SHIPWRECK_SPACING);
    int32_t regZ = floorDiv(constraint.chunkZ, SHIPWRECK_SPACING);
    rand.setRegionSeed(structureSeed, regX, regZ, SHIPWRECK_SALT);
    
    int32_t offset = SHIPWRECK_SPACING - SHIPWRECK_SEPARATION;
    if (regX * SHIPWRECK_SPACING + rand.nextInt(offset) != constraint.chunkX) return false;
    if (regZ * SHIPWRECK_SPACING + rand.nextInt(offset) != constraint.chunkZ) return false;

    rand.setCarverSeed(structureSeed, constraint.chunkX, constraint.chunkZ);
    if (static_cast<BlockRotation>(rand.nextInt(4)) != constraint.shipwreck.rotation) return false;
    
    ShipwreckType type;
    if (constraint.shipwreck.isBeached) {
        type = d_STRUCTURE_LOCATION_BEACHED[rand.nextInt(BEACHED_TYPE_COUNT)];
    } else {
        type = d_STRUCTURE_LOCATION_OCEAN[rand.nextInt(OCEAN_TYPE_COUNT)];
    }
    return type == constraint.shipwreck.type;
}

__device__ bool check_portal_full(int64_t structureSeed, const Constraint& constraint, StandaloneChunkRand& rand) {
    int32_t regX = floorDiv(constraint.chunkX, PORTAL_SPACING);
    int32_t regZ = floorDiv(constraint.chunkZ, PORTAL_SPACING);
    rand.setRegionSeed(structureSeed, regX, regZ, RUINED_PORTAL_SALT);

    int32_t offset = PORTAL_SPACING - PORTAL_SEPARATION;
    if (regX * PORTAL_SPACING + rand.nextInt(offset) != constraint.chunkX) return false;
    if (regZ * PORTAL_SPACING + rand.nextInt(offset) != constraint.chunkZ) return false;

    rand.setCarverSeed(structureSeed, constraint.chunkX, constraint.chunkZ);
    
    switch (constraint.portal.category) {
        case BiomeCategory::DESERT: break;
        case BiomeCategory::JUNGLE: rand.nextFloat(); break;
        case BiomeCategory::MOUNTAINS:
            if (rand.nextFloat() >= 0.5f) { rand.nextFloat(); }
            break;
    }

    if (rand.nextFloat() < 0.05f) { // Giant Portal
        if (static_cast<PortalType>(static_cast<int>(PortalType::GIANT_PORTAL_1) + rand.nextInt(3)) != constraint.portal.type) return false;
    } else { // Regular Portal
        if (static_cast<PortalType>(static_cast<int>(PortalType::PORTAL_1) + rand.nextInt(10)) != constraint.portal.type) return false;
    }

    if (static_cast<BlockRotation>(rand.nextInt(4)) != constraint.portal.rotation) return false;
    if (((rand.nextFloat() < 0.5f) ? BlockMirror::NONE : BlockMirror::FRONT_BACK) != constraint.portal.mirror) return false;
    
    return true;
}


// =======================================================================
// 5. Host-Side Abstraction Layer
// =======================================================================

// Base class for defining a searchable Minecraft structure.
class IStructure {
public:
    virtual ~IStructure() = default;
    virtual std::string get_name() const = 0;
    virtual ConstraintType get_type() const = 0;
    virtual bool try_parse_constraint(const std::vector<std::string>& parts, Constraint& out_constraint) const = 0;
    virtual void initialize_device_constants() const = 0;
    virtual bool has_fast_filter() const { return false; }
    virtual bool has_reversing_kernel() const { return false; }
};

// NEW: Concrete implementation for Villages.
class VillageStructure : public IStructure {
private:
    std::map<std::string, VillageStartPiece> name_to_piece;
    std::map<int, VillageType> biome_id_to_type;
public:
    VillageStructure() {
        // Map user biome IDs to internal enum
        biome_id_to_type[1] = VillageType::PLAINS;
        biome_id_to_type[2] = VillageType::SNOWY;
        biome_id_to_type[3] = VillageType::TAIGA;
        biome_id_to_type[4] = VillageType::SAVANNA;
        biome_id_to_type[5] = VillageType::DESERT;

        // Map piece names to internal enum
        name_to_piece["plains_fountain_01"] = VillageStartPiece::PLAINS_FOUNTAIN_01;
        name_to_piece["plains_meeting_point_1"] = VillageStartPiece::PLAINS_MEETING_POINT_1;
        name_to_piece["plains_meeting_point_2"] = VillageStartPiece::PLAINS_MEETING_POINT_2;
        name_to_piece["plains_meeting_point_3"] = VillageStartPiece::PLAINS_MEETING_POINT_3;
        name_to_piece["desert_meeting_point_1"] = VillageStartPiece::DESERT_MEETING_POINT_1;
        name_to_piece["desert_meeting_point_2"] = VillageStartPiece::DESERT_MEETING_POINT_2;
        name_to_piece["desert_meeting_point_3"] = VillageStartPiece::DESERT_MEETING_POINT_3;
        name_to_piece["savanna_meeting_point_1"] = VillageStartPiece::SAVANNA_MEETING_POINT_1;
        name_to_piece["savanna_meeting_point_2"] = VillageStartPiece::SAVANNA_MEETING_POINT_2;
        name_to_piece["savanna_meeting_point_3"] = VillageStartPiece::SAVANNA_MEETING_POINT_3;
        name_to_piece["savanna_meeting_point_4"] = VillageStartPiece::SAVANNA_MEETING_POINT_4;
        name_to_piece["taiga_meeting_point_1"] = VillageStartPiece::TAIGA_MEETING_POINT_1;
        name_to_piece["taiga_meeting_point_2"] = VillageStartPiece::TAIGA_MEETING_POINT_2;
        name_to_piece["snowy_meeting_point_1"] = VillageStartPiece::SNOWY_MEETING_POINT_1;
        name_to_piece["snowy_meeting_point_2"] = VillageStartPiece::SNOWY_MEETING_POINT_2;
        name_to_piece["snowy_meeting_point_3"] = VillageStartPiece::SNOWY_MEETING_POINT_3;
    }

    std::string get_name() const override { return "Village"; }
    ConstraintType get_type() const override { return ConstraintType::VILLAGE; }
    void initialize_device_constants() const override { /* Uses __constant__ memory directly */ }
    
    // Villages don't have a simple 20-bit filter or a reversible generation algorithm
    bool has_fast_filter() const override { return false; }
    bool has_reversing_kernel() const override { return false; }

    bool try_parse_constraint(const std::vector<std::string>& parts, Constraint& c) const override {
        // Format: ChunkX, ChunkZ, ROTATION, piece_name, biome_id, [is_abandoned]
        if (parts.size() < 5 || parts.size() > 6) return false;
        
        // Check if piece name and biome ID are valid for villages
        if (name_to_piece.find(parts[3]) == name_to_piece.end()) return false;
        int biome_id = -1;
        try { biome_id = std::stoi(parts[4]); } catch(...) { return false; }
        if (biome_id_to_type.find(biome_id) == biome_id_to_type.end()) return false;
        
        c.type = get_type();
        c.village.piece = name_to_piece.at(parts[3]);
        c.village.type = biome_id_to_type.at(biome_id);

        // Handle optional 'abandoned' flag
        c.village.is_abandoned = false;
        if (parts.size() == 6) {
            std::string abandoned_str = parts[5];
            std::transform(abandoned_str.begin(), abandoned_str.end(), abandoned_str.begin(), ::tolower);
            if (abandoned_str == "yes") {
                c.village.is_abandoned = true;
            } else if (abandoned_str != "no") {
                return false; // Invalid value for abandoned flag
            }
        }
        
        return true;
    }
};

// Concrete implementation for Shipwrecks.
class ShipwreckStructure : public IStructure {
private:
    std::map<std::string, ShipwreckType> name_to_type;
public:
    ShipwreckStructure() {
        name_to_type["rightsideup_backhalf"] = ShipwreckType::RIGHTSIDEUP_BACKHALF; name_to_type["rightsideup_backhalf_degraded"] = ShipwreckType::RIGHTSIDEUP_BACKHALF_DEGRADED;
        name_to_type["rightsideup_fronthalf"] = ShipwreckType::RIGHTSIDEUP_FRONTHALF; name_to_type["rightsideup_fronthalf_degraded"] = ShipwreckType::RIGHTSIDEUP_FRONTHALF_DEGRADED;
        name_to_type["rightsideup_full"] = ShipwreckType::RIGHTSIDEUP_FULL; name_to_type["rightsideup_full_degraded"] = ShipwreckType::RIGHTSIDEUP_FULL_DEGRADED;
        name_to_type["sideways_backhalf"] = ShipwreckType::SIDEWAYS_BACKHALF; name_to_type["sideways_backhalf_degraded"] = ShipwreckType::SIDEWAYS_BACKHALF_DEGRADED;
        name_to_type["sideways_fronthalf"] = ShipwreckType::SIDEWAYS_FRONTHALF; name_to_type["sideways_fronthalf_degraded"] = ShipwreckType::SIDEWAYS_FRONTHALF_DEGRADED;
        name_to_type["sideways_full"] = ShipwreckType::SIDEWAYS_FULL; name_to_type["sideways_full_degraded"] = ShipwreckType::SIDEWAYS_FULL_DEGRADED;
        name_to_type["upsidedown_backhalf"] = ShipwreckType::UPSIDEDOWN_BACKHALF; name_to_type["upsidedown_backhalf_degraded"] = ShipwreckType::UPSIDEDOWN_BACKHALF_DEGRADED;
        name_to_type["upsidedown_fronthalf"] = ShipwreckType::UPSIDEDOWN_FRONTHALF; name_to_type["upsidedown_fronthalf_degraded"] = ShipwreckType::UPSIDEDOWN_FRONTHALF_DEGRADED;
        name_to_type["upsidedown_full"] = ShipwreckType::UPSIDEDOWN_FULL; name_to_type["upsidedown_full_degraded"] = ShipwreckType::UPSIDEDOWN_FULL_DEGRADED;
        name_to_type["with_mast"] = ShipwreckType::WITH_MAST; name_to_type["with_mast_degraded"] = ShipwreckType::WITH_MAST_DEGRADED;
    }
    std::string get_name() const override { return "Shipwreck"; }
    ConstraintType get_type() const override { return ConstraintType::SHIPWRECK; }
    bool has_fast_filter() const override { return true; }
    bool has_reversing_kernel() const override { return true; }

    void initialize_device_constants() const override {
        std::vector<ShipwreckType> ocean_types = {
            name_to_type.at("with_mast"), name_to_type.at("upsidedown_full"), name_to_type.at("upsidedown_fronthalf"), name_to_type.at("upsidedown_backhalf"),
            name_to_type.at("sideways_full"), name_to_type.at("sideways_fronthalf"), name_to_type.at("sideways_backhalf"), name_to_type.at("rightsideup_full"),
            name_to_type.at("rightsideup_fronthalf"), name_to_type.at("rightsideup_backhalf"), name_to_type.at("with_mast_degraded"),
            name_to_type.at("upsidedown_full_degraded"), name_to_type.at("upsidedown_fronthalf_degraded"), name_to_type.at("upsidedown_backhalf_degraded"),
            name_to_type.at("sideways_full_degraded"), name_to_type.at("sideways_fronthalf_degraded"), name_to_type.at("sideways_backhalf_degraded"),
            name_to_type.at("rightsideup_full_degraded"), name_to_type.at("rightsideup_fronthalf_degraded"), name_to_type.at("rightsideup_backhalf_degraded")
        };
        std::vector<ShipwreckType> beached_types = {
            name_to_type.at("with_mast"), name_to_type.at("sideways_full"), name_to_type.at("sideways_fronthalf"), name_to_type.at("sideways_backhalf"),
            name_to_type.at("rightsideup_full"), name_to_type.at("rightsideup_fronthalf"), name_to_type.at("rightsideup_backhalf"),
            name_to_type.at("with_mast_degraded"), name_to_type.at("rightsideup_full_degraded"),
            name_to_type.at("rightsideup_fronthalf_degraded"), name_to_type.at("rightsideup_backhalf_degraded")
        };
        CUDA_CHECK(cudaMemcpyToSymbol(d_STRUCTURE_LOCATION_OCEAN, ocean_types.data(), ocean_types.size() * sizeof(ShipwreckType)));
        CUDA_CHECK(cudaMemcpyToSymbol(d_STRUCTURE_LOCATION_BEACHED, beached_types.data(), beached_types.size() * sizeof(ShipwreckType)));
    }

    bool try_parse_constraint(const std::vector<std::string>& parts, Constraint& c) const override {
        if (parts.size() != 5) return false;
        if (name_to_type.find(parts[3]) == name_to_type.end()) return false;

        c.type = get_type();
        c.shipwreck.type = name_to_type.at(parts[3]);
        std::string biome = parts[4];
        std::transform(biome.begin(), biome.end(), biome.begin(), [](unsigned char ch){ return std::tolower(ch); });
        if (biome == "beached") c.shipwreck.isBeached = true;
        else if (biome == "ocean") c.shipwreck.isBeached = false;
        else return false;

        return true;
    }
};

// Concrete implementation for Ruined Portals.
class RuinedPortalStructure : public IStructure {
private:
    std::map<std::string, PortalType> name_to_type;
public:
    RuinedPortalStructure() {
        name_to_type["portal_1"] = PortalType::PORTAL_1; name_to_type["portal_2"] = PortalType::PORTAL_2; name_to_type["portal_3"] = PortalType::PORTAL_3;
        name_to_type["portal_4"] = PortalType::PORTAL_4; name_to_type["portal_5"] = PortalType::PORTAL_5; name_to_type["portal_6"] = PortalType::PORTAL_6;
        name_to_type["portal_7"] = PortalType::PORTAL_7; name_to_type["portal_8"] = PortalType::PORTAL_8; name_to_type["portal_9"] = PortalType::PORTAL_9;
        name_to_type["portal_10"] = PortalType::PORTAL_10;
        name_to_type["giant_portal_1"] = PortalType::GIANT_PORTAL_1; name_to_type["giant_portal_2"] = PortalType::GIANT_PORTAL_2;
        name_to_type["giant_portal_3"] = PortalType::GIANT_PORTAL_3;
    }
    std::string get_name() const override { return "Ruined Portal"; }
    ConstraintType get_type() const override { return ConstraintType::RUINED_PORTAL; }
    bool has_reversing_kernel() const override { return true; }

    void initialize_device_constants() const override { /* No __constant__ memory for portals */ }

    bool try_parse_constraint(const std::vector<std::string>& parts, Constraint& c) const override {
        if (parts.size() != 6) return false;
        if (name_to_type.find(parts[3]) == name_to_type.end()) return false;

        c.type = get_type();
        c.portal.type = name_to_type.at(parts[3]);
        std::string mirror = parts[4];
        std::transform(mirror.begin(), mirror.end(), mirror.begin(), [](unsigned char ch){ return std::tolower(ch); });
        if (mirror == "yes") c.portal.mirror = BlockMirror::FRONT_BACK;
        else if (mirror == "no") c.portal.mirror = BlockMirror::NONE;
        else return false;

        int cat = std::stoi(parts[5]);
        if (cat < 1 || cat > 3) return false;
        c.portal.category = static_cast<BiomeCategory>(cat);

        return true;
    }
};

// Manages all known structure types and parsing logic.
class StructureRegistry {
private:
    std::vector<std::unique_ptr<IStructure>> structures;
    std::map<std::string, BlockRotation> name_to_rot;

public:
    StructureRegistry() {
        structures.push_back(std::make_unique<ShipwreckStructure>());
        structures.push_back(std::make_unique<RuinedPortalStructure>());
        structures.push_back(std::make_unique<VillageStructure>()); // NEW

        name_to_rot["NONE"] = BlockRotation::NONE; name_to_rot["CLOCKWISE_90"] = BlockRotation::CLOCKWISE_90;
        name_to_rot["CLOCKWISE_180"] = BlockRotation::CLOCKWISE_180; name_to_rot["COUNTERCLOCKWISE_90"] = BlockRotation::COUNTERCLOCKWISE_90;
    }

    void initialize_all_device_constants() const {
        for (const auto& s : structures) {
            s->initialize_device_constants();
        }
    }
    
    bool parse_line(const std::string& line, Constraint& out_constraint) const {
        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> parts;
        while(std::getline(ss, token, ',')) {
            token.erase(0, token.find_first_not_of(" \t"));
            token.erase(token.find_last_not_of(" \t") + 1);
            parts.push_back(token);
        }

        if (parts.size() < 4) return false;

        try {
            Constraint c;
            c.chunkX = std::stoi(parts[0]);
            c.chunkZ = std::stoi(parts[1]);
            if (name_to_rot.find(parts[2]) == name_to_rot.end()) return false;
            BlockRotation rot = name_to_rot.at(parts[2]);

            for (const auto& s : structures) {
                if (s->try_parse_constraint(parts, c)) {
                    if (c.type == ConstraintType::RUINED_PORTAL) c.portal.rotation = rot;
                    else if (c.type == ConstraintType::SHIPWRECK) c.shipwreck.rotation = rot;
                    else if (c.type == ConstraintType::VILLAGE) c.village.rotation = rot;
                    out_constraint = c;
                    return true;
                }
            }
        } catch (const std::exception&) {
            return false;
        }
        return false;
    }
};
// =======================================================================
// 5. CUDA Kernels (Global Scope)
// Kernels must be defined in the global namespace, not inside classes.
// =======================================================================

__global__ void pillarseedSearch_kernel(
    uint32_t pillarseed, const Constraint* d_constraints, int num_constraints,
    int64_t* d_found_seeds, uint32_t* d_found_count
) {
    uint32_t lowerbits = blockIdx.x;
    if (lowerbits >= 65536) return;

    int64_t partial_state = ((int64_t)pillarseed << 16) | lowerbits;
    int64_t state1 = partial_state * PILLAR_MULT + PILLAR_ADD;
    int64_t state2 = state1 * PILLAR_MULT + PILLAR_ADD;
    uint32_t half_seed_32bit = (uint32_t)((state2 ^ XOR_MASK) & 0xFFFFFFFFLL);

    StandaloneChunkRand rand;

    for (uint32_t upper16 = threadIdx.x; upper16 < 65536; upper16 += blockDim.x) {
        int64_t candidateSeed = ((int64_t)upper16 << 32) | half_seed_32bit;
        bool valid_for_all_constraints = true;
        for (int i = 0; i < num_constraints; i++) {
            bool ok;
            // MODIFIED: Added VILLAGE case
            if (d_constraints[i].type == ConstraintType::RUINED_PORTAL) {
                ok = check_portal_full(candidateSeed, d_constraints[i], rand);
            } else if (d_constraints[i].type == ConstraintType::SHIPWRECK) {
                ok = check_shipwreck_full(candidateSeed, d_constraints[i], rand);
            } else { // Village
                ok = check_village_full(candidateSeed, d_constraints[i], rand);
            }
            if (!ok) {
                valid_for_all_constraints = false;
                break;
            }
        }
        if (valid_for_all_constraints) {
            uint32_t result_idx = atomicAdd(d_found_count, 1);
            d_found_seeds[result_idx] = candidateSeed;
        }
    }
}

__device__ bool canGenerate_shipwreck_20bit_fast_filter(uint32_t lower20bits, int32_t chunkX, int32_t chunkZ) {
    int32_t regX = floorDiv(chunkX, SHIPWRECK_SPACING);
    int32_t regZ = floorDiv(chunkZ, SHIPWRECK_SPACING);
    uint32_t regionalSeed32 = (uint32_t)(((long long)lower20bits + (long long)regX * MULT_A + (long long)regZ * MULT_B + (long long)SHIPWRECK_SALT) ^ XOR_MASK);
    regionalSeed32 = (uint32_t)((long long)regionalSeed32 * LCG_MULT + LCG_ADD);
    uint32_t xCheck = (regionalSeed32 >> 17) & 3; 
    regionalSeed32 = (uint32_t)((long long)regionalSeed32 * LCG_MULT + LCG_ADD);
    uint32_t zCheck = (regionalSeed32 >> 17) & 3;
    return xCheck == (chunkX & 3) && zCheck == (chunkZ & 3);
}

__global__ void findLower20BitSeeds_kernel(const Constraint* d_shipwreck_constraints, int num_constraints, uint32_t* d_results, uint32_t* d_result_count) {
    uint32_t lower20bits = blockIdx.x * blockDim.x + threadIdx.x;
    if (lower20bits >= (1 << 20)) return;
    for (int i = 0; i < num_constraints; ++i) {
        if (!canGenerate_shipwreck_20bit_fast_filter(lower20bits, d_shipwreck_constraints[i].chunkX, d_shipwreck_constraints[i].chunkZ)) {
            return;
        }
    }
    uint32_t index = atomicAdd(d_result_count, 1);
    d_results[index] = lower20bits;
}

__global__ void reverseAndCheck_kernel(
    const uint32_t* d_valid_lower20bits, uint32_t num_valid_lower20bits,
    const Constraint* d_anchor, const Constraint* d_validators, int num_validators,
    int64_t* d_found_seeds, uint32_t* d_found_count
) {
    uint32_t lower20_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (lower20_idx >= num_valid_lower20bits) return;

    uint32_t lower20bit_seed = d_valid_lower20bits[lower20_idx];
    StandaloneChunkRand rand;
    
    // This is the validation check that is run on any found seed candidate.
    // It's a helper lambda to avoid code duplication.
    auto validate_candidate = [&](int64_t seed) -> bool {
        for(int i=0; i<num_validators; ++i) {
            bool ok;
            // MODIFIED: Added VILLAGE case
            if(d_validators[i].type == ConstraintType::RUINED_PORTAL) {
                ok = check_portal_full(seed, d_validators[i], rand);
            } else if (d_validators[i].type == ConstraintType::SHIPWRECK) {
                ok = check_shipwreck_full(seed, d_validators[i], rand);
            } else { // Village
                ok = check_village_full(seed, d_validators[i], rand);
            }
            if(!ok) return false;
        }
        return true;
    };


    if (d_anchor->type == ConstraintType::RUINED_PORTAL) {
        int32_t regX = floorDiv(d_anchor->chunkX, PORTAL_SPACING);
        int32_t regZ = floorDiv(d_anchor->chunkZ, PORTAL_SPACING);
        int32_t gen_regionsize = PORTAL_SPACING - PORTAL_SEPARATION;
        int32_t expectedRelX = ((d_anchor->chunkX % PORTAL_SPACING) + PORTAL_SPACING) % PORTAL_SPACING;
        int32_t expectedRelZ = ((d_anchor->chunkZ % PORTAL_SPACING) + PORTAL_SPACING) % PORTAL_SPACING;
        int64_t term_x = (int64_t)regX * MULT_A;
        int64_t term_z = (int64_t)regZ * MULT_B;
        uint64_t u_initial_part = (uint64_t)lower20bit_seed + (uint64_t)term_x + (uint64_t)term_z + (uint64_t)RUINED_PORTAL_SALT;
        uint64_t u_state0 = (u_initial_part ^ (uint64_t)XOR_MASK) & MASK_48;
        uint64_t u_state1 = (u_state0 * (uint64_t)LCG_MULT + (uint64_t)LCG_ADD) & MASK_48;
        uint64_t u_state2 = (u_state1 * (uint64_t)LCG_MULT + (uint64_t)LCG_ADD) & MASK_48;
        uint32_t lower20_of_state2 = (uint32_t)(u_state2 & 0xFFFFF);
        int32_t K = (lower20_of_state2 >> 17);
        int32_t R = expectedRelZ;
        int32_t i_base = (K - R % 8 + 8) % 8;
        int32_t B_base = 25 * i_base + R;
        for (int m = 0; ; ++m) {
            int64_t bits_z_64 = (int64_t)200 * m + B_base;
            if (bits_z_64 >= (1LL << 31)) break;
            int32_t bits_z = (int32_t)bits_z_64;
            if (bits_z - (bits_z % gen_regionsize) + (gen_regionsize - 1) >= 0) {
                uint64_t state2_candidate = (((uint64_t)bits_z) << 17) | (lower20_of_state2 & 0x1FFFF);
                uint64_t state1_candidate = ((state2_candidate - (uint64_t)LCG_ADD) * (uint64_t)LCG_MULT_INV) & MASK_48;
                int32_t bits_x = (int32_t)(state1_candidate >> 17);
                if (bits_x >= 0 && bits_x % gen_regionsize == expectedRelX && (bits_x - (bits_x % gen_regionsize) + (gen_regionsize - 1) >= 0)) {
                    uint64_t state0_candidate = ((state1_candidate - (uint64_t)LCG_ADD) * (uint64_t)LCG_MULT_INV) & MASK_48;
                    uint64_t u_scrambled = state0_candidate ^ (uint64_t)XOR_MASK;
                    int64_t seed = (int64_t)((u_scrambled - (uint64_t)term_x - (uint64_t)term_z - (uint64_t)RUINED_PORTAL_SALT) & MASK_48);
                    if (check_portal_full(seed, *d_anchor, rand) && validate_candidate(seed)) { 
                        d_found_seeds[atomicAdd(d_found_count, 1)] = seed; 
                    }
                }
            }
        }
    } else if (d_anchor->type == ConstraintType::SHIPWRECK) {
        int32_t regX = floorDiv(d_anchor->chunkX, SHIPWRECK_SPACING);
        int32_t regZ = floorDiv(d_anchor->chunkZ, SHIPWRECK_SPACING);
        int32_t gen_regionsize = SHIPWRECK_SPACING - SHIPWRECK_SEPARATION;
        int32_t expectedRelX = ((d_anchor->chunkX % SHIPWRECK_SPACING) + SHIPWRECK_SPACING) % SHIPWRECK_SPACING;
        int32_t expectedRelZ = ((d_anchor->chunkZ % SHIPWRECK_SPACING) + SHIPWRECK_SPACING) % SHIPWRECK_SPACING;
        int64_t term_x = (int64_t)regX * MULT_A;
        int64_t term_z = (int64_t)regZ * MULT_B;
        uint64_t u_initial_part = (uint64_t)lower20bit_seed + (uint64_t)term_x + (uint64_t)term_z + (uint64_t)SHIPWRECK_SALT;
        uint64_t u_state0 = (u_initial_part ^ (uint64_t)XOR_MASK) & MASK_48;
        uint64_t u_state1 = (u_state0 * (uint64_t)LCG_MULT + (uint64_t)LCG_ADD) & MASK_48;
        uint64_t u_state2 = (u_state1 * (uint64_t)LCG_MULT + (uint64_t)LCG_ADD) & MASK_48;
        uint32_t finalLower20LCG = (uint32_t)(u_state2 & 0xFFFFF);
        uint32_t base_Z_contrib = finalLower20LCG >> 17;
        for (uint32_t test_u = 0; test_u < 5; test_u++) {
            if ((((test_u << 3) + base_Z_contrib) % gen_regionsize) == expectedRelZ) {
                for (long j = 0; ; j++) {
                    uint64_t upper28LCG = 5 * j + test_u;
                    if (upper28LCG >= (1ULL << 28)) break;
                    uint64_t u_lcgStateForZ = (upper28LCG << 20) | finalLower20LCG;
                    uint64_t u_lcgStateForX = ((u_lcgStateForZ - (uint64_t)LCG_ADD) * (uint64_t)LCG_MULT_INV) & MASK_48;
                    if (((u_lcgStateForX >> 17) % gen_regionsize) == expectedRelX) {
                        uint64_t u_lcgInitial = ((u_lcgStateForX - (uint64_t)LCG_ADD) * (uint64_t)LCG_MULT_INV) & MASK_48;
                        uint64_t u_scrambled = u_lcgInitial ^ (uint64_t)XOR_MASK;
                        int64_t seed = (int64_t)((u_scrambled - (uint64_t)term_x - (uint64_t)term_z - (uint64_t)SHIPWRECK_SALT) & MASK_48);
                        if (check_shipwreck_full(seed, *d_anchor, rand) && validate_candidate(seed)) {
                            d_found_seeds[atomicAdd(d_found_count, 1)] = seed; 
                        }
                    }
                }
            }
        }
    }
}

__global__ void bruteforceStructureSeeds_kernel(
    const uint32_t* d_valid_lower20bits, uint32_t num_valid_lower20bits,
    const Constraint* d_constraints, int num_constraints,
    int64_t* d_found_seeds, uint32_t* d_found_count
) {
    uint64_t num_upper_bits_to_check = 1ULL << 28;
    uint64_t total_tasks = (uint64_t)num_valid_lower20bits * num_upper_bits_to_check;
    uint64_t thread_id = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;

    StandaloneChunkRand rand;

    for (uint64_t global_idx = thread_id; global_idx < total_tasks; global_idx += stride) {
        uint32_t lower20_idx = global_idx / num_upper_bits_to_check;
        uint32_t upper28_bits = global_idx % num_upper_bits_to_check;
        
        uint32_t lower20_val = d_valid_lower20bits[lower20_idx];
        int64_t candidateSeed = ((int64_t)upper28_bits << 20) | lower20_val;
        
        bool valid_for_all = true;
        for (int i = 0; i < num_constraints; i++) {
            bool ok;
            // MODIFIED: Added VILLAGE case
            if (d_constraints[i].type == ConstraintType::RUINED_PORTAL) {
                ok = check_portal_full(candidateSeed, d_constraints[i], rand);
            } else if (d_constraints[i].type == ConstraintType::SHIPWRECK) {
                ok = check_shipwreck_full(candidateSeed, d_constraints[i], rand);
            } else { // Village
                ok = check_village_full(candidateSeed, d_constraints[i], rand);
            }
            if (!ok) {
                valid_for_all = false;
                break;
            }
        }
        if (valid_for_all) {
            uint32_t result_idx = atomicAdd(d_found_count, 1);
            d_found_seeds[result_idx] = candidateSeed;
        }
    }
}

// =======================================================================
// 6. Search Strategy Classes
// =======================================================================

class ISearchStrategy {
public:
    virtual ~ISearchStrategy() = default;
    virtual void execute(CudaBuffer<int64_t>& results_buffer, CudaBuffer<uint32_t>& count_buffer) = 0;
    virtual std::string get_description() const = 0;
};

class PillarseedSearch : public ISearchStrategy {
private:
    uint32_t pillarseed;
    CudaBuffer<Constraint> d_constraints;

public:
    PillarseedSearch(uint32_t seed, const std::vector<Constraint>& constraints) : pillarseed(seed) {
        if (constraints.empty()) {
            throw std::runtime_error("Pillarseed mode requires at least one constraint.");
        }
        d_constraints = CudaBuffer<Constraint>(constraints.size());
        d_constraints.copy_to_device(constraints.data(), constraints.size());
    }
    
    std::string get_description() const override {
        std::stringstream ss;
        ss << "--- Pillarseed Mode Activated ---\n"
           << "Using Pillarseed: " << pillarseed << "\n"
           << "With " << d_constraints.size() << " constraint(s).\n\n"
           << "Launching kernel to check 2^32 full seeds...";
        return ss.str();
    }

    void execute(CudaBuffer<int64_t>& results, CudaBuffer<uint32_t>& count) override {
        int threads_per_block = 256;
        int blocks = 65536;
        pillarseedSearch_kernel<<<blocks, threads_per_block>>>(
            pillarseed, d_constraints.get(), d_constraints.size(), results.get(), count.get()
        );
    }
};

class StandardSearch : public ISearchStrategy {
private:
    std::vector<Constraint> all_constraints;
    CudaBuffer<uint32_t> d_valid_lower20bits;
    uint32_t lower20_count = 0;

public:
    StandardSearch(const std::vector<Constraint>& constraints) : all_constraints(constraints) {}
    
    std::string get_description() const override { return "--- Standard Structure Seed Search ---"; }
    
    void run_stage1() {
        std::cout << "\n--- Stage 1: Filtering Lower 20-bit seed patterns ---\n";
        std::vector<Constraint> shipwreck_constraints;
        bool has_fast_filterable_constraint = false;
        for(const auto& c : all_constraints) {
            if(c.type == ConstraintType::SHIPWRECK) {
                shipwreck_constraints.push_back(c);
                has_fast_filterable_constraint = true;
            }
        }

        if (has_fast_filterable_constraint) {
            std::cout << "Using " << shipwreck_constraints.size() << " shipwreck(s) to filter 20-bit candidates...\n";
            CudaBuffer<Constraint> d_shipwreck_constraints(shipwreck_constraints.size());
            d_shipwreck_constraints.copy_to_device(shipwreck_constraints.data(), shipwreck_constraints.size());
            
            CudaBuffer<uint32_t> d_lower20_results(1 << 20);
            CudaBuffer<uint32_t> d_lower20_count(1);
            d_lower20_count.memset(0);
            
            int threads = 256;
            int blocks = ((1 << 20) + threads - 1) / threads;
            findLower20BitSeeds_kernel<<<blocks, threads>>>(d_shipwreck_constraints.get(), shipwreck_constraints.size(), d_lower20_results.get(), d_lower20_count.get());
            
            d_lower20_count.copy_to_host(&lower20_count, 1);
            if (lower20_count > 0) {
                d_valid_lower20bits = CudaBuffer<uint32_t>(lower20_count);
                CUDA_CHECK(cudaMemcpy(d_valid_lower20bits.get(), d_lower20_results.get(), lower20_count * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
            }
        } else {
            std::cout << "No fast-filterable constraints (e.g., Shipwreck) provided. Generating all 2^20 candidates.\n";
            lower20_count = 1 << 20;
            std::vector<uint32_t> h_all_bits(lower20_count);
            std::iota(h_all_bits.begin(), h_all_bits.end(), 0);
            d_valid_lower20bits = CudaBuffer<uint32_t>(lower20_count);
            d_valid_lower20bits.copy_to_device(h_all_bits.data(), lower20_count);
        }
        
        std::cout << "Found " << lower20_count << " potential 20-bit candidates.\n";
    }
    
    void run_stage2_reversing(CudaBuffer<int64_t>& results, CudaBuffer<uint32_t>& count) {
        std::cout << "\n--- Stage 2: Using REVERSING Approach (1-2 constraints with reversible anchor) ---\n";
        int anchor_idx = -1;
        // Prioritize portal as anchor, then shipwreck
        for (int i = 0; i < all_constraints.size(); ++i) {
            if (all_constraints[i].type == ConstraintType::RUINED_PORTAL) { anchor_idx = i; break; }
        }
        if (anchor_idx == -1) {
            for (int i = 0; i < all_constraints.size(); ++i) {
                if (all_constraints[i].type == ConstraintType::SHIPWRECK) { anchor_idx = i; break; }
            }
        }

        Constraint h_anchor = all_constraints[anchor_idx];
        std::vector<Constraint> h_validators;
        for (int i = 0; i < all_constraints.size(); ++i) {
            if (i != anchor_idx) h_validators.push_back(all_constraints[i]);
        }
        
        CudaBuffer<Constraint> d_anchor(1);
        d_anchor.copy_to_device(&h_anchor, 1);
        CudaBuffer<Constraint> d_validators(h_validators.size());
        if (!h_validators.empty()) {
            d_validators.copy_to_device(h_validators.data(), h_validators.size());
        }

        const char* anchor_type_str = h_anchor.type == ConstraintType::RUINED_PORTAL ? "Portal" : "Shipwreck";
        std::cout << "Using " << anchor_type_str << " at [" << h_anchor.chunkX << "," << h_anchor.chunkZ << "] as anchor.\n";
        
        int threads = 256;
        int blocks = (lower20_count + threads - 1) / threads;
        reverseAndCheck_kernel<<<blocks, threads>>>(
            d_valid_lower20bits.get(), lower20_count, d_anchor.get(), d_validators.get(), 
            h_validators.size(), results.get(), count.get()
        );
    }
    
    void run_stage2_bruteforce(CudaBuffer<int64_t>& results, CudaBuffer<uint32_t>& count) {
        std::cout << "\n--- Stage 2: Using BRUTE-FORCE Approach (" << all_constraints.size() << " constraints) ---\n";
        CudaBuffer<Constraint> d_all_constraints(all_constraints.size());
        d_all_constraints.copy_to_device(all_constraints.data(), all_constraints.size());
        
        int threads = 256;
        int blocks = 32768;
        uint64_t total_tasks = (uint64_t)lower20_count * (1ULL << 28);
        std::cout << "Launching bruteforce kernel to check " << total_tasks << " total seed candidates...\n";

        bruteforceStructureSeeds_kernel<<<blocks, threads>>>(
            d_valid_lower20bits.get(), lower20_count, d_all_constraints.get(), 
            all_constraints.size(), results.get(), count.get()
        );
    }

    void execute(CudaBuffer<int64_t>& results, CudaBuffer<uint32_t>& count) override {
        run_stage1();
        if (lower20_count == 0) {
            std::cout << "No seed candidates found in Stage 1. Exiting.\n";
            return;
        }

        bool has_reversible_anchor = false;
        for (const auto& c : all_constraints) {
            if (c.type == ConstraintType::RUINED_PORTAL || c.type == ConstraintType::SHIPWRECK) {
                has_reversible_anchor = true;
                break;
            }
        }
        
        bool use_reversing = has_reversible_anchor && all_constraints.size() >= 1 && all_constraints.size() <= 10;
        if (use_reversing) {
            run_stage2_reversing(results, count);
        } else {
            run_stage2_bruteforce(results, count);
        }
    }
};

// =======================================================================
// 7. Main Application Logic
// =======================================================================

void print_usage(const char* prog_name) {
    std::cerr << "Usage: " << prog_name << " <constraints_file.txt>\n\n";
    std::cerr << "File Format (one constraint per line, '#' for comments):\n";
    std::cerr << "  - Structure-specific formats are detected automatically.\n";
    std::cerr << "  - Common Format: ChunkX, ChunkZ, ROTATION, ...specifics...\n\n";
    std::cerr << "Known Formats:\n";
    std::cerr << "  Shipwreck: ChunkX, ChunkZ, ROTATION, type_name, Ocean|Beached\n";
    std::cerr << "  Portal:    ChunkX, ChunkZ, ROTATION, portal_type, yes|no, biome_category(1-3)\n";
    std::cerr << "  Village:   ChunkX, ChunkZ, ROTATION, piece_name, biome_id, [yes|no]\n";
    std::cerr << "             -> biome_id: 1=Plains, 2=Snowy, 3=Taiga, 4=Savanna, 5=Desert\n";
    std::cerr << "             -> [yes|no] for is_abandoned is optional, defaults to 'no'.\n\n";
    std::cerr << "Pillarseed Mode:\n";
    std::cerr << "  Add the 32-bit pillarseed as the final number on its own line in the file.\n\n";
    std::cerr << "Example (Standard): -54, -14, COUNTERCLOCKWISE_90, sideways_fronthalf, Ocean\n";
    std::cerr << "Example (Village):  20, -112, NONE, taiga_meeting_point_1, 3, no\n";
}

bool load_constraints_from_file(const std::string& filename, const StructureRegistry& registry,
                                std::vector<Constraint>& constraints, int64_t& out_pillarseed) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file '" << filename << "'" << std::endl;
        return false;
    }

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(file, line)) {
        line.erase(0, line.find_first_not_of(" \t\n\r"));
        line.erase(line.find_last_not_of(" \t\n\r") + 1);
        if (line.empty() || line[0] == '#') continue;
        lines.push_back(line);
    }
    
    out_pillarseed = -1;

    if (!lines.empty()) {
        const std::string& last_line = lines.back();
        if (last_line.find(',') == std::string::npos) {
            try {
                size_t chars_processed;
                long long potential_seed = std::stoll(last_line, &chars_processed);
                if (chars_processed == last_line.length() && potential_seed >= 0 && potential_seed <= 0xFFFFFFFFLL) {
                    out_pillarseed = potential_seed;
                    lines.pop_back();
                }
            } catch (...) { /* Not a number */ }
        }
    }

    int line_num = 0;
    for (const auto& l : lines) {
        line_num++;
        Constraint c;
        if (registry.parse_line(l, c)) {
            constraints.push_back(c);
        } else {
            std::cerr << "Warning: Malformed or unknown constraint on line " << line_num << ": \"" << l << "\". Skipping." << std::endl;
        }
    }
    return !constraints.empty() || (out_pillarseed != -1);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    auto startTime = std::chrono::high_resolution_clock::now();
    
    StructureRegistry registry;
    registry.initialize_all_device_constants();
    
    int64_t h_pillarseed = -1;
    std::vector<Constraint> h_all_constraints;
    if (!load_constraints_from_file(argv[1], registry, h_all_constraints, h_pillarseed)) {
        std::cerr << "No valid constraints or pillarseed found in file. Exiting." << std::endl;
        return 1;
    }

    const uint32_t results_buffer_size = 20000000;
    CudaBuffer<int64_t> d_found_seeds(results_buffer_size);
    CudaBuffer<uint32_t> d_found_count(1);
    d_found_count.memset(0);

    std::unique_ptr<ISearchStrategy> strategy;
    try {
        if (h_pillarseed != -1) {
            strategy = std::make_unique<PillarseedSearch>(h_pillarseed, h_all_constraints);
        } else {
            strategy = std::make_unique<StandardSearch>(h_all_constraints);
        }
    } catch (const std::runtime_error& e) {
        std::cerr << "Error initializing search strategy: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << strategy->get_description() << std::endl;
    strategy->execute(d_found_seeds, d_found_count);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    uint32_t h_found_count = 0;
    d_found_count.copy_to_host(&h_found_count, 1);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    std::cout << "\n--- Search Complete in " << std::chrono::duration<double>(endTime - startTime).count() << " seconds ---\n";

    if (h_found_count == 0) {
        std::cout << "No structure seeds found.\n";
    } else {
        if (h_found_count > results_buffer_size) {
             std::cerr << "\nFATAL ERROR: Found " << h_found_count << " seeds, which exceeds buffer size of " 
                       << results_buffer_size << ". Results are incomplete." << std::endl;
             h_found_count = results_buffer_size;
        }
        std::cout << "Found " << h_found_count << " valid seed(s). Writing to found_seeds.txt...\n";
        std::vector<int64_t> h_found_seeds(h_found_count);
        d_found_seeds.copy_to_host(h_found_seeds.data(), h_found_count);
        
        std::sort(h_found_seeds.begin(), h_found_seeds.end());
        std::ofstream outfile("found_seeds.txt");
        for (const auto& seed : h_found_seeds) outfile << seed << "\n";
        outfile.close();
        std::cout << "Done.\n";
    }
    
    return 0;
}