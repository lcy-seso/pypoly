<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [Unification of concepts and constructs for deep learning programming](#unification-of-concepts-and-constructs-for-deep-learning-programming)
	- [Motivating examples](#motivating-examples)
		- [xx](#xx)
		- [Hierarchical sequences: complicated data-dependent control structures](#hierarchical-sequences-complicated-data-dependent-control-structures)
	- [Core Concepts](#core-concepts)
		- [Tensor](#tensor)
			- [Mutable Tensor](#mutable-tensor)
			- [Immutable Tensor](#immutable-tensor)
		- [Shape](#shape)
		- [Tensor Array](#tensor-array)
			- [_n_-dimensional Tensor Array](#n-dimensional-tensor-array)
		- [Function](#function)
	- [Constructs](#constructs)
		- [Looping construct](#looping-construct)

<!-- /TOC -->

# Unification of concepts and constructs for deep learning programming

## Motivating examples

### xx

### Hierarchical sequences: complicated data-dependent control structures


## Core Concepts

### Tensor

The minimum unit that is consumed in computation and is considered by our optimizations. Shape of a `Tensor` should be statically determined.

A tensor is a _n_-dimensional array of basic arithmetic types, including integer and real number types. In extreme cases, a tensor is degraded into a scalar.

Data flows among tensor operations will be analyzed.


#### Mutable Tensor

#### Immutable Tensor

### Shape

Shape is important since it affect memory layout analysis and memory management.

### Homogeneous Dynamic Tensor Array

Tensor Array provides a logical view of a _**compact continuous**_ memory.

An Array is a variable-length list of homogeneous elements. The length of an array can depend on data, but each element stored in the array must have a same type and the type is statically determined.

#### _n_-dimensional Tensor Array

_n_-dimensional Array is a nested Array. Elements of it is an Array.

### Function

## Constructs

### Looping construct

It return is a `Tensor Array` (?), whereas an imperative `for` loop has no returned value and uses side effects to convey the computation result. loop creates `Tensor Array` while loop nests create a high-dimensional `Tensor Array`. The initialization of the (high dimensional) `Tensor Array` is unnecessary.

A loop nest is binded to an nested TensorArray.

Iterate over the array and perform a user-defined computation.
