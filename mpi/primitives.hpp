/*
 * primitives.hpp
 *
 *  Created on: Dec 14, 2011
 *      Author: koji
 */

#ifndef PRIMITIVES_HPP_
#define PRIMITIVES_HPP_

#include "mpi_workarounds.h"

//-------------------------------------------------------------//
// Edge Types
//-------------------------------------------------------------//

struct UnweightedEdge;
template <> struct MpiTypeOf<UnweightedEdge> { static MPI_Datatype type; };
MPI_Datatype MpiTypeOf<UnweightedEdge>::type = MPI_DATATYPE_NULL;

struct UnweightedEdge {
	int64_t v0_;
	int64_t v1_;

	typedef int no_weight;

	int64_t v0() const { return v0_; }
	int64_t v1() const { return v1_; }
	void set(int64_t v0, int64_t v1) { v0_ = v0; v1_ = v1; }

	static void initialize()
	{
		int block_length[] = {1, 1};
		MPI_Aint displs[] = {
				reinterpret_cast<MPI_Aint>(&(static_cast<UnweightedEdge*>(NULL)->v0_)),
				reinterpret_cast<MPI_Aint>(&(static_cast<UnweightedEdge*>(NULL)->v1_)) };
		MPI_Type_create_hindexed(2, block_length, displs, MPI_INT64_T, &MpiTypeOf<UnweightedEdge>::type);
		MPI_Type_commit(&MpiTypeOf<UnweightedEdge>::type);
	}

	static void uninitialize()
	{
		MPI_Type_free(&MpiTypeOf<UnweightedEdge>::type);
	}
};

struct UnweightedPackedEdge;
template <> struct MpiTypeOf<UnweightedPackedEdge> { static MPI_Datatype type; };
MPI_Datatype MpiTypeOf<UnweightedPackedEdge>::type = MPI_DATATYPE_NULL;

struct UnweightedPackedEdge {
	uint32_t v0_low_;
	uint32_t v1_low_;
	uint32_t high_;

	typedef int no_weight;

	int64_t v0() const { return (v0_low_ | (static_cast<int64_t>(high_ & 0xFFFF) << 32)); }
	int64_t v1() const { return (v1_low_ | (static_cast<int64_t>(high_ >> 16) << 32)); }
	void set(int64_t v0, int64_t v1) {
		v0_low_ = static_cast<uint32_t>(v0);
		v1_low_ = static_cast<uint32_t>(v1);
		high_ = ((v0 >> 32) & 0xFFFF) | ((v1 >> 16) & 0xFFFF0000U);
	}

	static void initialize()
	{
		int block_length[] = {1, 1, 1};
		MPI_Aint displs[] = {
				reinterpret_cast<MPI_Aint>(&(static_cast<UnweightedPackedEdge*>(NULL)->v0_low_)),
				reinterpret_cast<MPI_Aint>(&(static_cast<UnweightedPackedEdge*>(NULL)->v1_low_)),
				reinterpret_cast<MPI_Aint>(&(static_cast<UnweightedPackedEdge*>(NULL)->high_)) };
		MPI_Type_create_hindexed(3, block_length, displs, MPI_UINT32_T, &MpiTypeOf<UnweightedPackedEdge>::type);
		MPI_Type_commit(&MpiTypeOf<UnweightedPackedEdge>::type);
	}

	static void uninitialize()
	{
		MPI_Type_free(&MpiTypeOf<UnweightedPackedEdge>::type);
	}
};

struct WeightedEdge;
template <> struct MpiTypeOf<WeightedEdge> { static MPI_Datatype type; };
MPI_Datatype MpiTypeOf<WeightedEdge>::type = MPI_DATATYPE_NULL;

struct WeightedEdge {
	uint32_t v0_low_;
	uint32_t v1_low_;
	uint32_t high_;
	int weight_;

	typedef int has_weight;

	int64_t v0() const { return (v0_low_ | (static_cast<int64_t>(high_ & 0xFFFF) << 32)); }
	int64_t v1() const { return (v1_low_ | (static_cast<int64_t>(high_ >> 16) << 32)); }
	int weight() const { return weight_; }
	void set(int64_t v0, int64_t v1) {
		v0_low_ = static_cast<uint32_t>(v0);
		v1_low_ = static_cast<uint32_t>(v1);
		high_ = ((v0 >> 32) & 0xFFFF) | ((v1 >> 16) & 0xFFFF0000U);
	}
	void set(int64_t v0, int64_t v1, int weight) {
		set(v0, v1);
		weight_ = weight;
	}

	static void initialize()
	{
		int block_length[] = {1, 1, 1, 1};
		MPI_Aint displs[] = {
				reinterpret_cast<MPI_Aint>(&(static_cast<WeightedEdge*>(NULL)->v0_low_)),
				reinterpret_cast<MPI_Aint>(&(static_cast<WeightedEdge*>(NULL)->v1_low_)),
				reinterpret_cast<MPI_Aint>(&(static_cast<WeightedEdge*>(NULL)->high_)),
				reinterpret_cast<MPI_Aint>(&(static_cast<WeightedEdge*>(NULL)->weight_)) };
		MPI_Datatype types[] = {MPI_UINT32_T, MPI_UINT32_T, MPI_UINT32_T, MPI_INT};
		MPI_Type_create_struct(4, block_length, displs, types, &MpiTypeOf<WeightedEdge>::type);
		MPI_Type_commit(&MpiTypeOf<WeightedEdge>::type);
	}

	static void uninitialize()
	{
		MPI_Type_free(&MpiTypeOf<WeightedEdge>::type);
	}
};

//-------------------------------------------------------------//
// Index Array Types
//-------------------------------------------------------------//

struct Rail40bitElement {
	uint32_t high[4];
	uint8_t low[4];
};

class Rail40bit {
public:
	static const int bytes_per_edge = 5;

	Rail40bit() : rail_(NULL) { }
	~Rail40bit() { this->free(); }
	void alloc(int64_t length) {
		int alloc_length = (length + 3) / 4;
		rail_ = static_cast<Rail40bitElement*>(cache_aligned_xmalloc(alloc_length*sizeof(Rail40bitElement)));
#ifndef NDEBUG
		memset(rail_, 0x00, alloc_length*sizeof(rail_[0]));
#endif
	}
	void free() { ::free(rail_); rail_ = NULL; }

	int64_t operator()(int64_t index) const {
		Rail40bitElement* elm = &rail_[index/4];
		return static_cast<int64_t>(elm->low[index%4]) |
				(static_cast<int64_t>(elm->high[index%4]) << 8);
	}
	uint8_t low_bits(int64_t index) const {
		Rail40bitElement* elm = &rail_[index/4];
		return elm->low[index%4];
	}
	void set(int64_t index, int64_t value) {
		Rail40bitElement* elm = &rail_[index/4];
		elm->low[index%4] = static_cast<uint8_t>(value);
		elm->high[index%4] = static_cast<int32_t>(value >> 8);
	}
	void move(int64_t to, int64_t from, int64_t size) {
		if(to < from) {
			for(int i = 0; i < size; ++i) {
				set(to+i, (*this)(from+i));
			}
		}
		else if(from < to) {
			for(int i = size; i > 0; --i) {
				set(to+i-1, (*this)(from+i-1));
			}
		}
	}
	void copy_from(int64_t to, Rail40bit& array, int64_t from, int64_t size) {
		for(int i = 0; i < size; ++i) {
			set(to+i, array(from+i));
		}
	}
private:
	Rail40bitElement *rail_;
};

class Pack48bit {
public:
	static const int bytes_per_edge = 6;

	Pack48bit() : i32_(NULL), i16_(NULL) { }
	~Pack48bit() { this->free(); }
	void alloc(int64_t length) {
		i32_ = static_cast<int32_t*>(cache_aligned_xmalloc(length*sizeof(int32_t)));
		i16_ = static_cast<uint16_t*>(cache_aligned_xmalloc(length*sizeof(uint16_t)));
#ifndef NDEBUG
		memset(i32_, 0x00, length*sizeof(i32_[0]));
		memset(i16_, 0x00, length*sizeof(i16_[0]));
#endif
	}
	void free() { ::free(i32_); i32_ = NULL; ::free(i16_); i16_ = NULL; }

	int64_t operator()(int64_t index) const {
		return static_cast<int64_t>(i16_[index]) |
				(static_cast<int64_t>(i32_[index]) << 16);
	}
	uint16_t low_bits(int64_t index) const { return i16_[index]; }
	void set(int64_t index, int64_t value) {
		i16_[index] = static_cast<uint16_t>(value);
		i32_[index] = static_cast<int32_t>(value >> 16);
	}
	void move(int64_t to, int64_t from, int64_t size) {
		memmove(i32_ + to, i32_ + from, sizeof(i32_[0])*size);
		memmove(i16_ + to, i16_ + from, sizeof(i16_[0])*size);
	}
	void copy_from(int64_t to, Pack48bit& array, int64_t from, int64_t size) {
		memcpy(i32_ + to, array.i32_ + from, sizeof(i32_[0])*size);
		memcpy(i16_ + to, array.i16_ + from, sizeof(i16_[0])*size);
	}

	int32_t* get_ptr_high() { return i32_; }
	uint16_t* get_ptr_low() { return i16_; }
private:
	int32_t *i32_;
	uint16_t *i16_;
};

class Rail48bit {
public:
	static const int bytes_per_edge = 6;

	Rail48bit() : rail_(NULL) { }
	~Rail48bit() { this->free(); }
	void alloc(int64_t length) {
		rail_ = static_cast<uint16_t*>(cache_aligned_xmalloc(length*sizeof(uint16_t)*3));
#ifndef NDEBUG
		memset(rail_, 0x00, length*sizeof(rail_[0])*3);
#endif
	}
	void free() { ::free(rail_); rail_ = NULL; }

	int64_t operator()(int64_t index) const {
		return   static_cast<int64_t>(rail_[index*3 + 0]) |
				(static_cast<int64_t>(rail_[index*3 + 1]) << 16) |
				(static_cast<int64_t>(rail_[index*3 + 2]) << 32);
	}
	uint16_t low_bits(int64_t index) const { return rail_[index*3]; }
	void set(int64_t index, int64_t value) {
		rail_[index*3 + 0] = static_cast<uint16_t>(value);
		rail_[index*3 + 1] = static_cast<uint16_t>(value >> 16);
		rail_[index*3 + 2] = static_cast<uint16_t>(value >> 32);
	}
	void move(int64_t to, int64_t from, int64_t size) {
		memmove(rail_ + to*3, rail_ + from*3, sizeof(rail_[0])*size*3);
	}
	void copy_from(int64_t to, Rail48bit& array, int64_t from, int64_t size) {
		memmove(rail_ + to*3, array.rail_ + from*3, sizeof(rail_[0])*size*3);
	}

	uint16_t* get_ptr() { return rail_; }
private:
	uint16_t *rail_;
};

class Rail32bit {
public:
	static const int bytes_per_edge = 4;

	Rail32bit() : rail_(NULL) { }
	~Rail32bit() { this->free(); }
	void alloc(int64_t length) {
		rail_ = static_cast<uint32_t*>(cache_aligned_xmalloc(length*sizeof(uint32_t)));
#ifndef NDEBUG
		memset(rail_, 0x00, length*sizeof(rail_[0]));
#endif
	}
	void free() { ::free(rail_); rail_ = NULL; }

	int64_t operator()(int64_t index) const {
		return   static_cast<int64_t>(rail_[index]);
	}
	uint16_t low_bits(int64_t index) const { return rail_[index]; }
	void set(int64_t index, int64_t value) {
		rail_[index] = static_cast<uint32_t>(value);
	}
	void move(int64_t to, int64_t from, int64_t size) {
		memmove(rail_ + to, rail_ + from, sizeof(rail_[0])*size);
	}
	void copy_from(int64_t to, Rail32bit& array, int64_t from, int64_t size) {
		memmove(rail_ + to, array.rail_ + from, sizeof(rail_[0])*size);
	}

	uint32_t* get_ptr() { return rail_; }
private:
	uint32_t *rail_;
};

class Pack64bit {
public:
	static const int bytes_per_edge = 8;

	Pack64bit() : i64_(NULL) { }
	~Pack64bit() { this->free(); }
	void alloc(int64_t length) {
		i64_ = static_cast<int64_t*>(cache_aligned_xmalloc(length*sizeof(int64_t)));
#ifndef NDEBUG
		memset(i64_, 0x00, length*sizeof(int64_t));
#endif
	}
	void free() { ::free(i64_); i64_ = NULL; }

	int64_t operator()(int64_t index) const { return i64_[index]; }
	int64_t low_bits(int64_t index) const { return i64_[index]; }
	void set(int64_t index, int64_t value) { i64_[index] = value; }
	void move(int64_t to, int64_t from, int64_t size) {
		memmove(i64_ + to, i64_ + from, sizeof(i64_[0])*size);
	}
	void copy_from(int64_t to, Pack64bit& array, int64_t from, int64_t size) {
		memcpy(i64_ + to, array.i64_ + from, sizeof(i64_[0])*size);
	}

	int64_t* get_ptr() { return i64_; }
private:
	int64_t *i64_;
};


#endif /* PRIMITIVES_HPP_ */
