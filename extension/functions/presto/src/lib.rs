// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use arrow::array::*;
use arrow::array::{as_list_array, ArrayRef, Float64Array};
use arrow::buffer::OffsetBuffer;
use arrow::compute;
use arrow::datatypes::DataType::List;
use arrow::datatypes::{DataType, Field};
use core::any::type_name;
use datafusion::error::Result;
use datafusion::logical_expr::Volatility;
use datafusion_common::cast::as_float64_array;
use datafusion_common::DataFusionError;
use datafusion_expr::{
    ReturnTypeFunction, ScalarFunctionDef, ScalarFunctionPackage, Signature,
};
use std::collections::HashSet;
use std::sync::Arc;

#[derive(Debug)]
pub struct AddOneFunction;

impl ScalarFunctionDef for AddOneFunction {
    fn name(&self) -> &str {
        "add_one"
    }

    fn signature(&self) -> Signature {
        Signature::exact(vec![DataType::Float64], Volatility::Immutable)
    }

    fn return_type(&self) -> ReturnTypeFunction {
        let return_type = Arc::new(DataType::Float64);
        Arc::new(move |_| Ok(return_type.clone()))
    }

    fn execute(&self, args: &[ArrayRef]) -> Result<ArrayRef> {
        assert_eq!(args.len(), 1);
        let input = as_float64_array(&args[0]).expect("cast failed");
        let array = input
            .iter()
            .map(|value| match value {
                Some(value) => Some(value + 1.0),
                _ => None,
            })
            .collect::<Float64Array>();
        Ok(Arc::new(array) as ArrayRef)
    }
}

#[derive(Debug)]
pub struct MultiplyTwoFunction;

impl ScalarFunctionDef for MultiplyTwoFunction {
    fn name(&self) -> &str {
        "multiply_two"
    }

    fn signature(&self) -> Signature {
        Signature::exact(vec![DataType::Float64], Volatility::Immutable)
    }

    fn return_type(&self) -> ReturnTypeFunction {
        let return_type = Arc::new(DataType::Float64);
        Arc::new(move |_| Ok(return_type.clone()))
    }

    fn execute(&self, args: &[ArrayRef]) -> Result<ArrayRef> {
        assert_eq!(args.len(), 1);
        let input = as_float64_array(&args[0]).expect("cast failed");
        let array = input
            .iter()
            .map(|value| match value {
                Some(value) => Some(value * 2.0),
                _ => None,
            })
            .collect::<Float64Array>();
        Ok(Arc::new(array) as ArrayRef)
    }
}

macro_rules! downcast_arg {
    ($ARG:expr, $ARRAY_TYPE:ident) => {{
        $ARG.as_any().downcast_ref::<$ARRAY_TYPE>().ok_or_else(|| {
            DataFusionError::Internal(format!(
                "could not cast to {}",
                type_name::<$ARRAY_TYPE>()
            ))
        })?
    }};
}

macro_rules! call_array_function {
    ($DATATYPE:expr, false) => {
        match $DATATYPE {
            DataType::Utf8 => array_function!(StringArray),
            DataType::LargeUtf8 => array_function!(LargeStringArray),
            DataType::Boolean => array_function!(BooleanArray),
            DataType::Float32 => array_function!(Float32Array),
            DataType::Float64 => array_function!(Float64Array),
            DataType::Int8 => array_function!(Int8Array),
            DataType::Int16 => array_function!(Int16Array),
            DataType::Int32 => array_function!(Int32Array),
            DataType::Int64 => array_function!(Int64Array),
            DataType::UInt8 => array_function!(UInt8Array),
            DataType::UInt16 => array_function!(UInt16Array),
            DataType::UInt32 => array_function!(UInt32Array),
            DataType::UInt64 => array_function!(UInt64Array),
            _ => unreachable!(),
        }
    };
    ($DATATYPE:expr, $INCLUDE_LIST:expr) => {{
        match $DATATYPE {
            // DataType::List(_) => array_function!(ListArray),
            DataType::Utf8 => array_function!(StringArray),
            DataType::LargeUtf8 => array_function!(LargeStringArray),
            DataType::Boolean => array_function!(BooleanArray),
            // DataType::Float32 => array_function!(Float32Array),
            // DataType::Float64 => array_function!(Float64Array),
            DataType::Int8 => array_function!(Int8Array),
            DataType::Int16 => array_function!(Int16Array),
            DataType::Int32 => array_function!(Int32Array),
            DataType::Int64 => array_function!(Int64Array),
            DataType::UInt8 => array_function!(UInt8Array),
            DataType::UInt16 => array_function!(UInt16Array),
            DataType::UInt32 => array_function!(UInt32Array),
            DataType::UInt64 => array_function!(UInt64Array),
            _ => unreachable!(),
        }
    }};
}
// ---
// function:
//   name: abs
//   section: math
//   kind: scalar
//   parameters:
//     - datatype: decimal(p, s)
//   returns:
//     datatype: decimal(p, s)
// ---
// function:
//   name: abs
//   section: math
//   kind: scalar
//   parameters:
//     - datatype: double
//   returns:
//     datatype: double
// ---
// function:
//   name: abs
//   section: math
//   kind: scalar
//   parameters:
//     - datatype: bigint
//   returns:
//     datatype: int
// ---
// function:
//   name: abs
//   section: math
//   kind: scalar
//   parameters:
//     - datatype: float
//   returns:
//     datatype: float
// ---
// function:
//   name: abs
//   section: math
//   kind: scalar
//   parameters:
//     - datatype: smallint
//   returns:
//     datatype: smallint
// ---
// function:
//   name: abs
//   section: math
//   kind: scalar
//   parameters:
//     - datatype: tinyint
//   returns:
//     datatype: tinyint
// ---
// function:
//   name: acos
//   section: math
//   kind: scalar
//   parameters:
//     - datatype: double
//   returns:
//     datatype: double
// --- TODO
// function:
//   name: all_match
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: array(t)
//     - datatype: lambda <(t) -> boolean>
//   returns:
//     datatype: boolean
// ---
// function:
//   name: any_match
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: array(t)
//     - datatype: lambda <(t) -> boolean>
//   returns:
//     datatype: boolean
// --- TODO
// function:
//   name: approx_distinct
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: boolean
//   returns:
//     datatype: bigint
// ---
// function:
//   name: approx_distinct
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: boolean
//     - datatype: double
//   returns:
//     datatype: bigint
// ---
// function:
//   name: approx_distinct
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: t
//   returns:
//     datatype: bigint
// ---
// function:
//   name: approx_distinct
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: t
//     - datatype: double
//   returns:
//     datatype: bigint
// ---
// function:
//   name: approx_distinct
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: unknown
//   returns:
//     datatype: bigint
// ---
// function:
//   name: approx_distinct
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: unknown
//     - datatype: double
//   returns:
//     datatype: bigint
// --- TODO
// function:
//   name: approx_most_frequent
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: bigint
//     - datatype: bigint
//     - datatype: bigint
//   returns:
//     datatype: map<bigint, bigint>
// --- TODO
// function:
//   name: approx_most_frequent
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: bigint
//     - datatype: varchar
//     - datatype: bigint
//   returns:
//     datatype: map<varchar, bigint>
// ---
// function:
//   name: approx_percentile
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: bigint
//     - datatype: array<double>
//   returns:
//     datatype: array<bigint>
// ---
// function:
//   name: approx_percentile
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: bigint
//     - datatype: double
//     - datatype: array<double>
//   returns:
//     datatype: array<bigint>
// ---
// function:
//   name: approx_percentile
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: double
//     - datatype: array<double>
//   returns:
//     datatype: array<double>
// ---
// function:
//   name: approx_percentile
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: double
//     - datatype: double
//     - datatype: array<double>
//   returns:
//     datatype: array<double>
// ---
// function:
//   name: approx_percentile
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: float
//     - datatype: array<double>
//   returns:
//     datatype: array<float>
// ---
// function:
//   name: approx_percentile
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: float
//     - datatype: double
//     - datatype: array<double>
//   returns:
//     datatype: array<float>
// ---
// function:
//   name: approx_percentile
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: bigint
//     - datatype: double
//   returns:
//     datatype: bigint
// ---
// function:
//   name: approx_percentile
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: bigint
//     - datatype: double
//     - datatype: double
//   returns:
//     datatype: bigint
// ---
// function:
//   name: approx_percentile
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: bigint
//     - datatype: double
//     - datatype: double
//     - datatype: double
//   returns:
//     datatype: bigint
// ---
// function:
//   name: approx_percentile
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: double
//     - datatype: double
//   returns:
//     datatype: double
// ---
// function:
//   name: approx_percentile
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: double
//     - datatype: double
//     - datatype: double
//   returns:
//     datatype: double
// ---
// function:
//   name: approx_percentile
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: double
//     - datatype: double
//     - datatype: double
//     - datatype: double
//   returns:
//     datatype: double
// ---
// function:
//   name: approx_percentile
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: float
//     - datatype: double
//   returns:
//     datatype: float
// ---
// function:
//   name: approx_percentile
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: float
//     - datatype: double
//     - datatype: double
//   returns:
//     datatype: float
// ---
// function:
//   name: approx_percentile
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: float
//     - datatype: double
//     - datatype: double
//     - datatype: double
//   returns:
//     datatype: float
// ---
// function:
//   name: approx_set
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: bigint
//   returns:
//     datatype: bytea
// ---
// function:
//   name: approx_set
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: double
//   returns:
//     datatype: bytea
// ---
// function:
//   name: approx_set
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: bytea
// ---
// function:
//   name: arbitrary
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: t
//   returns:
//     datatype: t
// ---
// function:
//   name: array_agg
//   section: aggregate
//   kind: aggregate
//   parameters:
//     - datatype: t
//   returns:
//     datatype: array(t)
// ---
// function:
//   name: array_distinct
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: array(e)
//   returns:
//     datatype: array(e)
#[derive(Debug)]
pub struct ArrayDistinctFunction;

impl ScalarFunctionDef for ArrayDistinctFunction {
    fn name(&self) -> &str {
        "array_distinct"
    }

    fn signature(&self) -> Signature {
        Signature::variadic_any(Volatility::Immutable)
    }

    fn return_type(&self) -> ReturnTypeFunction {
        Arc::new(move |tys| match &tys[0] {
            List(field) => Ok(Arc::new(List(Arc::new(Field::new(
                "item",
                field.data_type().clone(),
                true,
            ))))),
            _ => Err(DataFusionError::Internal(format!("todo"))),
        })
    }

    fn execute(&self, args: &[ArrayRef]) -> Result<ArrayRef> {
        assert_eq!(args.len(), 1);
        let arr = as_list_array(&args[0]);

        macro_rules! array_function {
            ($ARRAY_TYPE:ident) => {{
                let mut offsets: Vec<i32> = vec![0];
                let mut values =
                    downcast_arg!(new_empty_array(arr.data_type()), $ARRAY_TYPE).clone();

                for arr in arr.iter() {
                    let last_offset: i32 = offsets.last().copied().ok_or_else(|| {
                        DataFusionError::Internal(format!("offsets should not be empty"))
                    })?;
                    match arr {
                        Some(arr) => {
                            let child_array = downcast_arg!(arr, $ARRAY_TYPE);
                            let mut seen = HashSet::new();

                            let filter_array = child_array
                                .iter()
                                .map(|element| {
                                    if seen.contains(&element) {
                                        seen.insert(element);
                                        Some(false)
                                    } else {
                                        Some(true)
                                    }
                                })
                                .collect::<BooleanArray>();

                            let filtered_array =
                                compute::filter(&child_array, &filter_array)?;
                            values = downcast_arg!(
                                compute::concat(&[&values, &filtered_array,])?.clone(),
                                $ARRAY_TYPE
                            )
                            .clone();
                            offsets.push(last_offset + filtered_array.len() as i32);
                        }
                        None => offsets.push(last_offset),
                    }
                }

                let field = Arc::new(Field::new("item", arr.data_type().clone(), true));

                Arc::new(ListArray::try_new(
                    field,
                    OffsetBuffer::new(offsets.into()),
                    Arc::new(values),
                    None,
                )?)
            }};
        }
        let res = call_array_function!(arr.value_type(), true);
        Ok(res)
    }
}

// ---
// function:
//   name: array_except
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: array(e)
//     - datatype: array(e)
//   returns:
//     datatype: array(e)
// ---
// function:
//   name: array_intersect
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: array(e)
//     - datatype: array(e)
//   returns:
//     datatype: array(e)
// ---
// function:
//   name: array_join
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: array(t)
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: array_join
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: array(t)
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: array_max
//   section: math
//   kind: scalar
//   parameters:
//     - datatype: array(t)
//   returns:
//     datatype: t
// ---
// function:
//   name: array_min
//   section: math
//   kind: scalar
//   parameters:
//     - datatype: array(t)
//   returns:
//     datatype: t
// ---
// function:
//   name: array_position
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: array(t)
//     - datatype: t
//   returns:
//     datatype: bigint
// ---
// function:
//   name: array_remove
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: array(e)
//     - datatype: e
//   returns:
//     datatype: array(e)
// ---
// function:
//   name: array_sort
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: array(e)
//   returns:
//     datatype: array(e)
// ---
// function:
//   name: array_sort
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: array(t)
//     - datatype: lambda <(t, t) -> bigint>
//   returns:
//     datatype: array(t)
// ---
// function:
//   name: array_union
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: array(e)
//     - datatype: array(e)
//   returns:
//     datatype: array(e)
// ---
// function:
//   name: arrays_overlap
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: array(e)
//     - datatype: array(e)
//   returns:
//     datatype: boolean
// ---
// function:
//   name: asin
//   section: math
//   kind: scalar
//   parameters:
//     - datatype: double
//   returns:
//     datatype: double
// ---
// function:
//   name: at_timezone
//   section: temporal
//   kind: scalar
//   parameters:
//     - datatype: timestamp(p)
//     - datatype: varchar
//   returns:
//     datatype: timestamp(p)
// ---
// function:
//   name: atan
//   section: math
//   kind: scalar
//   parameters:
//     - datatype: double
//   returns:
//     datatype: double
// ---
// function:
//   name: atan2
//   section: math
//   kind: scalar
//   parameters:
//     - datatype: double
//     - datatype: double
//   returns:
//     datatype: double
// ---
// function:
//   name: avg
//   section: math
//   kind: aggregate
//   parameters:
//     - datatype: decimal(p, s)
//   returns:
//     datatype: decimal(p, s)
// ---
// function:
//   name: avg
//   section: math
//   kind: aggregate
//   parameters:
//     - datatype: bigint
//   returns:
//     datatype: double
// ---
// function:
//   name: avg
//   section: math
//   kind: aggregate
//   parameters:
//     - datatype: double
//   returns:
//     datatype: double
// ---
// function:
//   name: avg
//   section: math
//   kind: aggregate
//   parameters:
//     - datatype: interval day to second
//   returns:
//     datatype: interval day to second
// ---
// function:
//   name: avg
//   section: math
//   kind: aggregate
//   parameters:
//     - datatype: interval day to second
//   returns:
//     datatype: interval day to second
// ---
// function:
//   name: avg
//   section: math
//   kind: aggregate
//   parameters:
//     - datatype: float
//   returns:
//     datatype: float
// ---
// function:
//   name: bar
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: double
//     - datatype: bigint
//   returns:
//     datatype: varchar
// ---
// function:
//   name: bar
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: double
//     - datatype: bigint
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: beta_cdf
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: double
//     - datatype: double
//     - datatype: double
//   returns:
//     datatype: double
// ---
// function:
//   name: bing_tile
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: bigint
//     - datatype: bigint
//     - datatype: bigint
//   returns:
//     datatype: varchar
// ---
// function:
//   name: bing_tile
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: bing_tile_at
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: double
//     - datatype: double
//     - datatype: bigint
//   returns:
//     datatype: varchar
// ---
// function:
//   name: bing_tile_coordinates
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: struct<x int, y int>
// ---
// function:
//   name: bing_tile_polygon
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: bing_tile_quadkey
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: bing_tile_zoom_level
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: tinyint
// ---
// function:
//   name: bing_tiles_around
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: double
//     - datatype: double
//     - datatype: bigint
//   returns:
//     datatype: array<varchar>
// ---
// function:
//   name: bing_tiles_around
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: double
//     - datatype: double
//     - datatype: bigint
//     - datatype: double
//   returns:
//     datatype: array<varchar>
// ---
// function:
//   name: bit_count
//   section: aggregate
//   kind: scalar
//   parameters:
//     - datatype: bigint
//     - datatype: bigint
//   returns:
//     datatype: bigint
// ---
// function:
//   name: bitwise_and
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: bigint
//     - datatype: bigint
//   returns:
//     datatype: bigint
// ---
// function:
//   name: bitwise_and_agg
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: bigint
//   returns:
//     datatype: bigint
// ---
// function:
//   name: bitwise_left_shift
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: bigint
//     - datatype: bigint
//   returns:
//     datatype: bigint
// ---
// function:
//   name: bitwise_left_shift
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: int
//     - datatype: bigint
//   returns:
//     datatype: int
// ---
// function:
//   name: bitwise_left_shift
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: smallint
//     - datatype: bigint
//   returns:
//     datatype: smallint
// ---
// function:
//   name: bitwise_left_shift
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: tinyint
//     - datatype: bigint
//   returns:
//     datatype: tinyint
// ---
// function:
//   name: bitwise_not
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: bigint
//   returns:
//     datatype: bigint
// ---
// function:
//   name: bitwise_or
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: bigint
//     - datatype: bigint
//   returns:
//     datatype: bigint
// ---
// function:
//   name: bitwise_or_agg
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: bigint
//   returns:
//     datatype: bigint
// ---
// function:
//   name: bitwise_right_shift
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: bigint
//     - datatype: bigint
//   returns:
//     datatype: bigint
// ---
// function:
//   name: bitwise_right_shift
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: int
//     - datatype: bigint
//   returns:
//     datatype: int
// ---
// function:
//   name: bitwise_right_shift
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: smallint
//     - datatype: bigint
//   returns:
//     datatype: smallint
// ---
// function:
//   name: bitwise_right_shift
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: tinyint
//     - datatype: bigint
//   returns:
//     datatype: tinyint
// ---
// function:
//   name: bitwise_right_shift_arithmetic
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: bigint
//     - datatype: bigint
//   returns:
//     datatype: bigint
// ---
// function:
//   name: bitwise_right_shift_arithmetic
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: int
//     - datatype: bigint
//   returns:
//     datatype: int
// ---
// function:
//   name: bitwise_right_shift_arithmetic
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: smallint
//     - datatype: bigint
//   returns:
//     datatype: smallint
// ---
// function:
//   name: bitwise_right_shift_arithmetic
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: tinyint
//     - datatype: bigint
//   returns:
//     datatype: tinyint
// ---
// function:
//   name: bitwise_xor
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: bigint
//     - datatype: bigint
//   returns:
//     datatype: bigint
// ---
// function:
//   name: bool_and
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: boolean
//   returns:
//     datatype: boolean
// ---
// function:
//   name: bool_or
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: boolean
//   returns:
//     datatype: boolean
// ---
// function:
//   name: cardinality
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: array(e)
//   returns:
//     datatype: bigint
// ---
// function:
//   name: cardinality
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: bytea
//   returns:
//     datatype: bigint
// ---
// function:
//   name: cardinality
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: map(k, v)
//   returns:
//     datatype: bigint
// ---
// function:
//   name: cardinality
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: bigint
// ---
// function:
//   name: cbrt
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: double
//   returns:
//     datatype: double
// ---
// function:
//   name: ceil
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: decimal(p, s)
//   returns:
//     datatype: decimal(rp, 0)
// ---
// function:
//   name: ceil
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: double
//   returns:
//     datatype: double
// ---
// function:
//   name: ceil
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: float
//   returns:
//     datatype: float
// ---
// function:
//   name: ceiling
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: decimal(p, s)
//   returns:
//     datatype: decimal(rp, 0)
// ---
// function:
//   name: ceiling
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: double
//   returns:
//     datatype: double
// ---
// function:
//   name: ceiling
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: float
//   returns:
//     datatype: float
// ---
// function:
//   name: char2hexint
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: checksum
//   section: math
//   kind: aggregate
//   parameters:
//     - datatype: t
//   returns:
//     datatype: bytea
// ---
// function:
//   name: chr
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: bigint
//   returns:
//     datatype: varchar
// ---
// function:
//   name: classify
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: map<bigint, double>
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: codepoint
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: int
// ---
// function:
//   name: color
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: double
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: color
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: double
//     - datatype: double
//     - datatype: double
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: color
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: combinations
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: array(t)
//     - datatype: bigint
//   returns:
//     datatype: array(array(t))
// ---
// function:
//   name: concat
//   section: string
//   kind: scalar
//   parameters:
//     - datatype: e
//     - datatype: array(e)
//   returns:
//     datatype: array(e)
// ---
// function:
//   name: concat
//   section: string
//   variadic: uniform
//   kind: scalar
//   parameters:
//     - datatype: array(e)
//   returns:
//     datatype: array(e)
// ---
// function:
//   name: concat
//   section: string
//   kind: scalar
//   parameters:
//     - datatype: array(e)
//     - datatype: e
//   returns:
//     datatype: array(e)
// ---
// function:
//   name: concat
//   section: string
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: concat
//   section: string
//   kind: scalar
//   parameters:
//     - datatype: bytea
//   returns:
//     datatype: bytea
// ---
// function:
//   name: concat
//   section: string
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: concat_ws
//   section: string
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: array<varchar>
//   returns:
//     datatype: varchar
// ---
// function:
//   name: concat_ws
//   section: string
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: contains
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: array(t)
//     - datatype: t
//   returns:
//     datatype: boolean
// ---
// function:
//   name: contains
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: boolean
// ---
// function:
//   name: contains_sequence
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: array(t)
//     - datatype: array(t)
//   returns:
//     datatype: boolean
// ---
// function:
//   name: convex_hull_agg
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: corr
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: double
//     - datatype: double
//   returns:
//     datatype: double
// ---
// function:
//   name: corr
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: float
//     - datatype: float
//   returns:
//     datatype: float
// ---
// function:
//   name: cos
//   section: math
//   kind: scalar
//   parameters:
//     - datatype: double
//   returns:
//     datatype: double
// ---
// function:
//   name: cosh
//   section: math
//   kind: scalar
//   parameters:
//     - datatype: double
//   returns:
//     datatype: double
// ---
// function:
//   name: cosine_similarity
//   section: math
//   kind: scalar
//   parameters:
//     - datatype: map<varchar, double>
//     - datatype: map<varchar, double>
//   returns:
//     datatype: double
// ---
// function:
//   name: count
//   section: aggregate
//   kind: aggregate
//   parameters: []
//   returns:
//     datatype: bigint
// ---
// function:
//   name: count
//   section: aggregate
//   kind: aggregate
//   parameters:
//     - datatype: t
//   returns:
//     datatype: bigint
// ---
// function:
//   name: count_if
//   section: aggregate
//   kind: aggregate
//   parameters:
//     - datatype: boolean
//   returns:
//     datatype: bigint
// ---
// function:
//   name: covar_pop
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: double
//     - datatype: double
//   returns:
//     datatype: double
// ---
// function:
//   name: covar_pop
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: float
//     - datatype: float
//   returns:
//     datatype: float
// ---
// function:
//   name: covar_samp
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: double
//     - datatype: double
//   returns:
//     datatype: double
// ---
// function:
//   name: covar_samp
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: float
//     - datatype: float
//   returns:
//     datatype: float
// ---
// function:
//   name: crc32
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: bytea
//   returns:
//     datatype: bigint
// ---
// function:
//   name: cume_dist
//   section: other
//   kind: window
//   parameters: []
//   returns:
//     datatype: double
// ---
// function:
//   name: current_date
//   section: temporal
//   kind: scalar
//   parameters: []
//   returns:
//     datatype: date
// ---
// function:
//   name: current_groups
//   section: other
//   kind: scalar
//   parameters: []
//   returns:
//     datatype: array<varchar>
// ---
// function:
//   name: current_timezone
//   section: temporal
//   kind: scalar
//   parameters: []
//   returns:
//     datatype: varchar
// ---
// function:
//   name: date
//   section: temporal
//   kind: scalar
//   parameters:
//     - datatype: timestamp(p)
//   returns:
//     datatype: date
// ---
// function:
//   name: date
//   section: temporal
//   kind: scalar
//   parameters:
//     - datatype: timestamp(p)
//   returns:
//     datatype: date
// ---
// function:
//   name: date
//   section: temporal
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: date
// ---
// function:
//   name: date_add
//   section: temporal
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: bigint
//     - datatype: date
//   returns:
//     datatype: date
// ---
// function:
//   name: date_add
//   section: temporal
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: bigint
//     - datatype: time(p)
//   returns:
//     datatype: time(p)
// ---
// function:
//   name: date_add
//   section: temporal
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: bigint
//     - datatype: time(p)
//   returns:
//     datatype: time(p)
// ---
// function:
//   name: date_add
//   section: temporal
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: bigint
//     - datatype: timestamp(p)
//   returns:
//     datatype: timestamp(p)
// ---
// function:
//   name: date_add
//   section: temporal
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: bigint
//     - datatype: timestamp(p)
//   returns:
//     datatype: timestamp(p)
// ---
// function:
//   name: date_diff
//   section: temporal
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: date
//     - datatype: date
//   returns:
//     datatype: bigint
// ---
// function:
//   name: date_diff
//   section: temporal
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: time(p)
//     - datatype: time(p)
//   returns:
//     datatype: bigint
// ---
// function:
//   name: date_diff
//   section: temporal
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: time(p)
//     - datatype: time(p)
//   returns:
//     datatype: bigint
// ---
// function:
//   name: date_diff
//   section: temporal
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: timestamp(p)
//     - datatype: timestamp(p)
//   returns:
//     datatype: bigint
// ---
// function:
//   name: date_diff
//   section: temporal
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: timestamp(p)
//     - datatype: timestamp(p)
//   returns:
//     datatype: bigint
// ---
// function:
//   name: date_format
//   section: temporal
//   kind: scalar
//   parameters:
//     - datatype: timestamp(p)
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: date_format
//   section: temporal
//   kind: scalar
//   parameters:
//     - datatype: timestamp(p)
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: date_parse
//   section: temporal
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: timestamp(millisecond)
// ---
// function:
//   name: date_trunc
//   section: temporal
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: date
//   returns:
//     datatype: date
// ---
// function:
//   name: date_trunc
//   section: temporal
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: time(p)
//   returns:
//     datatype: time(p)
// ---
// function:
//   name: date_trunc
//   section: temporal
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: time(p)
//   returns:
//     datatype: time(p)
// ---
// function:
//   name: date_trunc
//   section: temporal
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: timestamp(p)
//   returns:
//     datatype: timestamp(p)
// ---
// function:
//   name: date_trunc
//   section: temporal
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: timestamp(p)
//   returns:
//     datatype: timestamp(p)
// ---
// function:
//   name: day
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: date
//   returns:
//     datatype: bigint
// ---
// function:
//   name: day
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: interval day to second
//   returns:
//     datatype: bigint
// ---
// function:
//   name: day
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: timestamp(p)
//   returns:
//     datatype: bigint
// ---
// function:
//   name: day
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: timestamp(p)
//   returns:
//     datatype: bigint
// ---
// function:
//   name: day_of_month
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: date
//   returns:
//     datatype: bigint
// ---
// function:
//   name: day_of_month
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: interval day to second
//   returns:
//     datatype: bigint
// ---
// function:
//   name: day_of_month
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: timestamp(p)
//   returns:
//     datatype: bigint
// ---
// function:
//   name: day_of_month
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: timestamp(p)
//   returns:
//     datatype: bigint
// ---
// function:
//   name: day_of_week
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: date
//   returns:
//     datatype: bigint
// ---
// function:
//   name: day_of_week
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: timestamp(p)
//   returns:
//     datatype: bigint
// ---
// function:
//   name: day_of_week
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: timestamp(p)
//   returns:
//     datatype: bigint
// ---
// function:
//   name: day_of_year
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: date
//   returns:
//     datatype: bigint
// ---
// function:
//   name: day_of_year
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: timestamp(p)
//   returns:
//     datatype: bigint
// ---
// function:
//   name: day_of_year
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: timestamp(p)
//   returns:
//     datatype: bigint
// ---
// function:
//   name: degrees
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: double
//   returns:
//     datatype: double
// ---
// function:
//   name: dense_rank
//   section: other
//   kind: window
//   parameters: []
//   returns:
//     datatype: bigint
// ---
// function:
//   name: dow
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: date
//   returns:
//     datatype: bigint
// ---
// function:
//   name: dow
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: timestamp(p)
//   returns:
//     datatype: bigint
// ---
// function:
//   name: dow
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: timestamp(p)
//   returns:
//     datatype: bigint
// ---
// function:
//   name: doy
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: date
//   returns:
//     datatype: bigint
// ---
// function:
//   name: doy
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: timestamp(p)
//   returns:
//     datatype: bigint
// ---
// function:
//   name: doy
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: timestamp(p)
//   returns:
//     datatype: bigint
// ---
// function:
//   name: e
//   section: other
//   kind: scalar
//   parameters: []
//   returns:
//     datatype: double
// ---
// function:
//   name: element_at
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: map(k, v)
//     - datatype: k
//   returns:
//     datatype: v
// ---
// function:
//   name: element_at
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: array(e)
//     - datatype: bigint
//   returns:
//     datatype: e
// ---
// function:
//   name: empty_approx_set
//   section: other
//   kind: scalar
//   parameters: []
//   returns:
//     datatype: bytea
// ---
// function:
//   name: evaluate_classifier_predictions
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: bigint
//     - datatype: bigint
//   returns:
//     datatype: varchar
// ---
// function:
//   name: evaluate_classifier_predictions
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: every
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: boolean
//   returns:
//     datatype: boolean
// ---
// function:
//   name: exp
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: double
//   returns:
//     datatype: double
// ---
// function:
//   name: features
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: double
//   returns:
//     datatype: map<bigint, double>
// ---
// function:
//   name: features
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: double
//     - datatype: double
//   returns:
//     datatype: map<bigint, double>
// ---
// function:
//   name: features
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: double
//     - datatype: double
//     - datatype: double
//   returns:
//     datatype: map<bigint, double>
// ---
// function:
//   name: features
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: double
//     - datatype: double
//     - datatype: double
//     - datatype: double
//   returns:
//     datatype: map<bigint, double>
// ---
// function:
//   name: features
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: double
//     - datatype: double
//     - datatype: double
//     - datatype: double
//     - datatype: double
//   returns:
//     datatype: map<bigint, double>
// ---
// function:
//   name: features
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: double
//     - datatype: double
//     - datatype: double
//     - datatype: double
//     - datatype: double
//     - datatype: double
//   returns:
//     datatype: map<bigint, double>
// ---
// function:
//   name: features
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: double
//     - datatype: double
//     - datatype: double
//     - datatype: double
//     - datatype: double
//     - datatype: double
//     - datatype: double
//   returns:
//     datatype: map<bigint, double>
// ---
// function:
//   name: features
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: double
//     - datatype: double
//     - datatype: double
//     - datatype: double
//     - datatype: double
//     - datatype: double
//     - datatype: double
//     - datatype: double
//   returns:
//     datatype: map<bigint, double>
// ---
// function:
//   name: features
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: double
//     - datatype: double
//     - datatype: double
//     - datatype: double
//     - datatype: double
//     - datatype: double
//     - datatype: double
//     - datatype: double
//     - datatype: double
//   returns:
//     datatype: map<bigint, double>
// ---
// function:
//   name: features
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: double
//     - datatype: double
//     - datatype: double
//     - datatype: double
//     - datatype: double
//     - datatype: double
//     - datatype: double
//     - datatype: double
//     - datatype: double
//     - datatype: double
//   returns:
//     datatype: map<bigint, double>
// ---
// function:
//   name: filter
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: array(t)
//     - datatype: lambda <(t) -> boolean>
//   returns:
//     datatype: array(t)
// ---
// function:
//   name: first_value
//   section: other
//   kind: window
//   parameters:
//     - datatype: t
//   returns:
//     datatype: t
// ---
// function:
//   name: flatten
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: array(array(e))
//   returns:
//     datatype: array(e)
// ---
// function:
//   name: floor
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: decimal(p, s)
//   returns:
//     datatype: decimal(rp, 0)
// ---
// function:
//   name: floor
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: double
//   returns:
//     datatype: double
// ---
// function:
//   name: floor
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: float
//   returns:
//     datatype: float
// ---
// function:
//   name: format_datetime
//   section: temporal
//   kind: scalar
//   parameters:
//     - datatype: timestamp(p)
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: format_datetime
//   section: temporal
//   kind: scalar
//   parameters:
//     - datatype: timestamp(p)
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: format_number
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: bigint
//   returns:
//     datatype: varchar
// ---
// function:
//   name: format_number
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: double
//   returns:
//     datatype: varchar
// ---
// function:
//   name: from_base
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: bigint
//   returns:
//     datatype: bigint
// ---
// function:
//   name: from_base32
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: bytea
//   returns:
//     datatype: bytea
// ---
// function:
//   name: from_base32
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: bytea
// ---
// function:
//   name: from_base64
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: bytea
//   returns:
//     datatype: bytea
// ---
// function:
//   name: from_base64
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: bytea
// ---
// function:
//   name: from_base64url
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: bytea
//   returns:
//     datatype: bytea
// ---
// function:
//   name: from_base64url
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: bytea
// ---
// function:
//   name: from_big_endian_32
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: bytea
//   returns:
//     datatype: int
// ---
// function:
//   name: from_big_endian_64
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: bytea
//   returns:
//     datatype: bigint
// ---
// function:
//   name: from_encoded_polyline
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: from_geojson_geometry
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: from_hex
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: bytea
//   returns:
//     datatype: bytea
// ---
// function:
//   name: from_hex
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: bytea
// ---
// function:
//   name: from_ieee754_32
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: bytea
//   returns:
//     datatype: float
// ---
// function:
//   name: from_ieee754_64
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: bytea
//   returns:
//     datatype: double
// ---
// function:
//   name: from_iso8601_date
//   section: temporal
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: date
// ---
// function:
//   name: from_iso8601_timestamp
//   section: temporal
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: timestamp(millisecond)
// ---
// function:
//   name: from_iso8601_timestamp_nanos
//   section: temporal
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: timestamp
// ---
// function:
//   name: from_unixtime
//   section: temporal
//   kind: scalar
//   parameters:
//     - datatype: double
//   returns:
//     datatype: timestamp(millisecond)
// ---
// function:
//   name: from_unixtime
//   section: temporal
//   kind: scalar
//   parameters:
//     - datatype: double
//     - datatype: bigint
//     - datatype: bigint
//   returns:
//     datatype: timestamp(millisecond)
// ---
// function:
//   name: from_unixtime
//   section: temporal
//   kind: scalar
//   parameters:
//     - datatype: double
//     - datatype: varchar
//   returns:
//     datatype: timestamp(millisecond)
// ---
// function:
//   name: from_unixtime_nanos
//   section: temporal
//   kind: scalar
//   parameters:
//     - datatype: bigint
//   returns:
//     datatype: timestamp
// ---
// function:
//   name: from_unixtime_nanos
//   section: temporal
//   kind: scalar
//   parameters:
//     - datatype: decimal(p, s)
//   returns:
//     datatype: timestamp
// ---
// function:
//   name: from_utf8
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: bytea
//   returns:
//     datatype: varchar
// ---
// function:
//   name: from_utf8
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: bytea
//     - datatype: bigint
//   returns:
//     datatype: varchar
// ---
// function:
//   name: from_utf8
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: bytea
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: geometric_mean
//   section: math
//   kind: aggregate
//   parameters:
//     - datatype: bigint
//   returns:
//     datatype: double
// ---
// function:
//   name: geometric_mean
//   section: math
//   kind: aggregate
//   parameters:
//     - datatype: double
//   returns:
//     datatype: double
// ---
// function:
//   name: geometric_mean
//   section: math
//   kind: aggregate
//   parameters:
//     - datatype: float
//   returns:
//     datatype: float
// ---
// function:
//   name: geometry_from_hadoop_shape
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: bytea
//   returns:
//     datatype: varchar
// ---
// function:
//   name: geometry_invalid_reason
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: geometry_nearest_points
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: struct<c0 varchar, c1 varchar>
// ---
// function:
//   name: geometry_to_bing_tiles
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: bigint
//   returns:
//     datatype: array<varchar>
// ---
// function:
//   name: geometry_union
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: array<varchar>
//   returns:
//     datatype: varchar
// ---
// function:
//   name: geometry_union_agg
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: great_circle_distance
//   section: math
//   kind: scalar
//   parameters:
//     - datatype: double
//     - datatype: double
//     - datatype: double
//     - datatype: double
//   returns:
//     datatype: double
// ---
// function:
//   name: greatest
//   section: other
//   variadic: uniform
//   kind: scalar
//   parameters:
//     - datatype: e
//   returns:
//     datatype: e
// ---
// function:
//   name: hamming_distance
//   section: math
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: bigint
// ---
// function:
//   name: hash_counts
//   section: aggregate
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: map<bigint, smallint>
// ---
// function:
//   name: histogram
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: t
//   returns:
//     datatype: map(t, bigint)
// ---
// function:
//   name: hmac_md5
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: bytea
//     - datatype: bytea
//   returns:
//     datatype: bytea
// ---
// function:
//   name: hmac_sha1
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: bytea
//     - datatype: bytea
//   returns:
//     datatype: bytea
// ---
// function:
//   name: hmac_sha256
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: bytea
//     - datatype: bytea
//   returns:
//     datatype: bytea
// ---
// function:
//   name: hmac_sha512
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: bytea
//     - datatype: bytea
//   returns:
//     datatype: bytea
// ---
// function:
//   name: hour
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: interval day to second
//   returns:
//     datatype: bigint
// ---
// function:
//   name: hour
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: time(p)
//   returns:
//     datatype: bigint
// ---
// function:
//   name: hour
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: time(p)
//   returns:
//     datatype: bigint
// ---
// function:
//   name: hour
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: timestamp(p)
//   returns:
//     datatype: bigint
// ---
// function:
//   name: hour
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: timestamp(p)
//   returns:
//     datatype: bigint
// ---
// function:
//   name: human_readable_seconds
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: double
//   returns:
//     datatype: varchar
// ---
// function:
//   name: index
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: bigint
// ---
// function:
//   name: infinity
//   section: other
//   kind: scalar
//   parameters: []
//   returns:
//     datatype: double
// ---
// function:
//   name: intersection_cardinality
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: bigint
// ---
// function:
//   name: inverse_beta_cdf
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: double
//     - datatype: double
//     - datatype: double
//   returns:
//     datatype: double
// ---
// function:
//   name: inverse_normal_cdf
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: double
//     - datatype: double
//     - datatype: double
//   returns:
//     datatype: double
// ---
// function:
//   name: is_finite
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: double
//   returns:
//     datatype: boolean
// ---
// function:
//   name: is_infinite
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: double
//   returns:
//     datatype: boolean
// ---
// function:
//   name: is_json_scalar
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: boolean
// ---
// function:
//   name: is_json_scalar
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: boolean
// ---
// function:
//   name: is_nan
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: double
//   returns:
//     datatype: boolean
// ---
// function:
//   name: is_nan
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: float
//   returns:
//     datatype: boolean
// ---
// function:
//   name: jaccard_index
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: double
// ---
// function:
//   name: json_array_contains
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: bigint
//   returns:
//     datatype: boolean
// ---
// function:
//   name: json_array_contains
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: boolean
//   returns:
//     datatype: boolean
// ---
// function:
//   name: json_array_contains
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: double
//   returns:
//     datatype: boolean
// ---
// function:
//   name: json_array_contains
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: boolean
// ---
// function:
//   name: json_array_contains
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: bigint
//   returns:
//     datatype: boolean
// ---
// function:
//   name: json_array_contains
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: boolean
//   returns:
//     datatype: boolean
// ---
// function:
//   name: json_array_contains
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: double
//   returns:
//     datatype: boolean
// ---
// function:
//   name: json_array_contains
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: boolean
// ---
// function:
//   name: json_array_get
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: bigint
//   returns:
//     datatype: varchar
// ---
// function:
//   name: json_array_get
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: bigint
//   returns:
//     datatype: varchar
// ---
// function:
//   name: json_array_length
//   section: string
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: bigint
// ---
// function:
//   name: json_array_length
//   section: string
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: bigint
// ---
// function:
//   name: json_extract
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: json_extract
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: json_extract_scalar
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: json_extract_scalar
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: json_format
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: json_parse
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: json_size
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: bigint
// ---
// function:
//   name: json_size
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: bigint
// ---
// function:
//   name: kurtosis
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: bigint
//   returns:
//     datatype: double
// ---
// function:
//   name: kurtosis
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: double
//   returns:
//     datatype: double
// ---
// function:
//   name: lag
//   section: other
//   kind: window
//   parameters:
//     - datatype: t
//   returns:
//     datatype: t
// ---
// function:
//   name: lag
//   section: other
//   kind: window
//   parameters:
//     - datatype: t
//     - datatype: bigint
//   returns:
//     datatype: t
// ---
// function:
//   name: lag
//   section: other
//   kind: window
//   parameters:
//     - datatype: t
//     - datatype: bigint
//     - datatype: t
//   returns:
//     datatype: t
// ---
// function:
//   name: last_day_of_month
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: date
//   returns:
//     datatype: date
// ---
// function:
//   name: last_day_of_month
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: timestamp(p)
//   returns:
//     datatype: date
// ---
// function:
//   name: last_day_of_month
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: timestamp(p)
//   returns:
//     datatype: date
// ---
// function:
//   name: last_value
//   section: other
//   kind: window
//   parameters:
//     - datatype: t
//   returns:
//     datatype: t
// ---
// function:
//   name: lead
//   section: other
//   kind: window
//   parameters:
//     - datatype: t
//   returns:
//     datatype: t
// ---
// function:
//   name: lead
//   section: other
//   kind: window
//   parameters:
//     - datatype: t
//     - datatype: bigint
//   returns:
//     datatype: t
// ---
// function:
//   name: lead
//   section: other
//   kind: window
//   parameters:
//     - datatype: t
//     - datatype: bigint
//     - datatype: t
//   returns:
//     datatype: t
// ---
// function:
//   name: learn_classifier
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: bigint
//     - datatype: map<bigint, double>
//   returns:
//     datatype: varchar
// ---
// function:
//   name: learn_classifier
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: double
//     - datatype: map<bigint, double>
//   returns:
//     datatype: varchar
// ---
// function:
//   name: learn_classifier
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: varchar
//     - datatype: map<bigint, double>
//   returns:
//     datatype: varchar
// ---
// function:
//   name: learn_libsvm_classifier
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: bigint
//     - datatype: map<bigint, double>
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: learn_libsvm_classifier
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: double
//     - datatype: map<bigint, double>
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: learn_libsvm_classifier
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: varchar
//     - datatype: map<bigint, double>
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: learn_libsvm_regressor
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: bigint
//     - datatype: map<bigint, double>
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: learn_libsvm_regressor
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: double
//     - datatype: map<bigint, double>
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: learn_regressor
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: bigint
//     - datatype: map<bigint, double>
//   returns:
//     datatype: varchar
// ---
// function:
//   name: learn_regressor
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: double
//     - datatype: map<bigint, double>
//   returns:
//     datatype: varchar
// ---
// function:
//   name: least
//   section: other
//   variadic: uniform
//   kind: scalar
//   parameters:
//     - datatype: e
//   returns:
//     datatype: e
// ---
// function:
//   name: length
//   section: string
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: bigint
// ---
// function:
//   name: length
//   section: string
//   kind: scalar
//   parameters:
//     - datatype: bytea
//   returns:
//     datatype: bigint
// ---
// function:
//   name: length
//   section: string
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: bigint
// ---
// function:
//   name: levenshtein_distance
//   section: math
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: bigint
// ---
// function:
//   name: line_interpolate_point
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: double
//   returns:
//     datatype: varchar
// ---
// function:
//   name: line_interpolate_points
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: double
//   returns:
//     datatype: array<varchar>
// ---
// function:
//   name: line_locate_point
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: double
// ---
// function:
//   name: listagg
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//     - datatype: boolean
//     - datatype: varchar
//     - datatype: boolean
//   returns:
//     datatype: varchar
// ---
// function:
//   name: ln
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: double
//   returns:
//     datatype: double
// ---
// function:
//   name: log
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: double
//     - datatype: double
//   returns:
//     datatype: double
// ---
// function:
//   name: log10
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: double
//   returns:
//     datatype: double
// ---
// function:
//   name: log2
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: double
//   returns:
//     datatype: double
// ---
// function:
//   name: lower
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: lower
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: lpad
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: bytea
//     - datatype: bigint
//     - datatype: bytea
//   returns:
//     datatype: bytea
// ---
// function:
//   name: lpad
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: bigint
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: ltrim
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: ltrim
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: ltrim
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: ltrim
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: luhn_check
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: boolean
// ---
// function:
//   name: make_set_digest
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: t
//   returns:
//     datatype: varchar
// ---
// function:
//   name: map
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: array(k)
//     - datatype: array(v)
//   returns:
//     datatype: map(k, v)
// ---
// function:
//   name: map
//   section: other
//   kind: scalar
//   parameters: []
//   returns:
//     datatype: map(unknown, unknown)
// ---
// function:
//   name: map_agg
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: k
//     - datatype: v
//   returns:
//     datatype: map(k, v)
// ---
// function:
//   name: map_concat
//   section: string
//   variadic: uniform
//   kind: scalar
//   parameters:
//     - datatype: map(k, v)
//   returns:
//     datatype: map(k, v)
// ---
// function:
//   name: map_entries
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: map(k, v)
//   returns:
//     datatype: array(row(c0 k, c1 v))
// ---
// function:
//   name: map_filter
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: map(k, v)
//     - datatype: lambda <(k, v) -> boolean>
//   returns:
//     datatype: map(k, v)
// ---
// function:
//   name: map_from_entries
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: array(row(c0 k, c1 v))
//   returns:
//     datatype: map(k, v)
// ---
// function:
//   name: map_keys
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: map(k, v)
//   returns:
//     datatype: array(k)
// ---
// function:
//   name: map_union
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: map(k, v)
//   returns:
//     datatype: map(k, v)
// ---
// function:
//   name: map_values
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: map(k, v)
//   returns:
//     datatype: array(v)
// ---
// function:
//   name: map_zip_with
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: map(k, v1)
//     - datatype: map(k, v2)
//     - datatype: lambda <(k, v1, v2) -> v3>
//   returns:
//     datatype: map(k, v3)
// ---
// function:
//   name: max
//   section: math
//   kind: aggregate
//   parameters:
//     - datatype: e
//     - datatype: bigint
//   returns:
//     datatype: array(e)
// ---
// function:
//   name: max
//   section: math
//   kind: aggregate
//   parameters:
//     - datatype: t
//   returns:
//     datatype: t
// ---
// function:
//   name: max_by
//   section: math
//   kind: aggregate
//   parameters:
//     - datatype: v
//     - datatype: k
//     - datatype: bigint
//   returns:
//     datatype: array(v)
// ---
// function:
//   name: max_by
//   section: math
//   kind: aggregate
//   parameters:
//     - datatype: v
//     - datatype: k
//   returns:
//     datatype: v
// ---
// function:
//   name: md5
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: bytea
//   returns:
//     datatype: bytea
// ---
// function:
//   name: merge
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: bytea
//   returns:
//     datatype: bytea
// ---
// function:
//   name: merge
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: merge
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: merge_set_digest
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: millisecond
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: interval day to second
//   returns:
//     datatype: bigint
// ---
// function:
//   name: millisecond
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: time(p)
//   returns:
//     datatype: bigint
// ---
// function:
//   name: millisecond
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: time(p)
//   returns:
//     datatype: bigint
// ---
// function:
//   name: millisecond
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: timestamp(p)
//   returns:
//     datatype: bigint
// ---
// function:
//   name: millisecond
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: timestamp(p)
//   returns:
//     datatype: bigint
// ---
// function:
//   name: min
//   section: math
//   kind: aggregate
//   parameters:
//     - datatype: e
//     - datatype: bigint
//   returns:
//     datatype: array(e)
// ---
// function:
//   name: min
//   section: math
//   kind: aggregate
//   parameters:
//     - datatype: t
//   returns:
//     datatype: t
// ---
// function:
//   name: min_by
//   section: math
//   kind: aggregate
//   parameters:
//     - datatype: v
//     - datatype: k
//     - datatype: bigint
//   returns:
//     datatype: array(v)
// ---
// function:
//   name: min_by
//   section: math
//   kind: aggregate
//   parameters:
//     - datatype: v
//     - datatype: k
//   returns:
//     datatype: v
// ---
// function:
//   name: minute
//   section: math
//   kind: scalar
//   parameters:
//     - datatype: interval day to second
//   returns:
//     datatype: bigint
// ---
// function:
//   name: minute
//   section: math
//   kind: scalar
//   parameters:
//     - datatype: time(p)
//   returns:
//     datatype: bigint
// ---
// function:
//   name: minute
//   section: math
//   kind: scalar
//   parameters:
//     - datatype: time(p)
//   returns:
//     datatype: bigint
// ---
// function:
//   name: minute
//   section: math
//   kind: scalar
//   parameters:
//     - datatype: timestamp(p)
//   returns:
//     datatype: bigint
// ---
// function:
//   name: minute
//   section: math
//   kind: scalar
//   parameters:
//     - datatype: timestamp(p)
//   returns:
//     datatype: bigint
// ---
// function:
//   name: mod
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: bigint
//     - datatype: bigint
//   returns:
//     datatype: bigint
// ---
// function:
//   name: mod
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: decimal(a_precision, a_scale)
//     - datatype: decimal(b_precision, b_scale)
//   returns:
//     datatype: decimal(r_precision, r_scale)
// ---
// function:
//   name: mod
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: double
//     - datatype: double
//   returns:
//     datatype: double
// ---
// function:
//   name: mod
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: int
//     - datatype: int
//   returns:
//     datatype: int
// ---
// function:
//   name: mod
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: float
//     - datatype: float
//   returns:
//     datatype: float
// ---
// function:
//   name: mod
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: smallint
//     - datatype: smallint
//   returns:
//     datatype: smallint
// ---
// function:
//   name: mod
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: tinyint
//     - datatype: tinyint
//   returns:
//     datatype: tinyint
// ---
// function:
//   name: month
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: date
//   returns:
//     datatype: bigint
// ---
// function:
//   name: month
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: interval day to second
//   returns:
//     datatype: bigint
// ---
// function:
//   name: month
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: timestamp(p)
//   returns:
//     datatype: bigint
// ---
// function:
//   name: month
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: timestamp(p)
//   returns:
//     datatype: bigint
// ---
// function:
//   name: multimap_agg
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: k
//     - datatype: v
//   returns:
//     datatype: map(k, array(v))
// ---
// function:
//   name: multimap_from_entries
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: array(row(c0 k, c1 v))
//   returns:
//     datatype: map(k, array(v))
// ---
// function:
//   name: murmur3
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: bytea
//   returns:
//     datatype: bytea
// ---
// function:
//   name: nan
//   section: other
//   kind: scalar
//   parameters: []
//   returns:
//     datatype: double
// ---
// function:
//   name: ngrams
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: array(t)
//     - datatype: bigint
//   returns:
//     datatype: array(array(t))
// ---
// function:
//   name: none_match
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: array(t)
//     - datatype: lambda <(t) -> boolean>
//   returns:
//     datatype: boolean
// ---
// function:
//   name: normal_cdf
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: double
//     - datatype: double
//     - datatype: double
//   returns:
//     datatype: double
// ---
// function:
//   name: normalize
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: now
//   section: other
//   kind: scalar
//   parameters: []
//   returns:
//     datatype: timestamp(millisecond)
// ---
// function:
//   name: nth_value
//   section: other
//   kind: window
//   parameters:
//     - datatype: t
//     - datatype: bigint
//   returns:
//     datatype: t
// ---
// function:
//   name: ntile
//   section: other
//   kind: window
//   parameters:
//     - datatype: bigint
//   returns:
//     datatype: bigint
// ---
// function:
//   name: numeric_histogram
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: bigint
//     - datatype: double
//   returns:
//     datatype: map<double, double>
// ---
// function:
//   name: numeric_histogram
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: bigint
//     - datatype: double
//     - datatype: double
//   returns:
//     datatype: map<double, double>
// ---
// function:
//   name: numeric_histogram
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: bigint
//     - datatype: float
//   returns:
//     datatype: map<float, float>
// ---
// function:
//   name: numeric_histogram
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: bigint
//     - datatype: float
//     - datatype: double
//   returns:
//     datatype: map<float, float>
// ---
// function:
//   name: objectid
//   section: other
//   kind: scalar
//   parameters: []
//   returns:
//     datatype: varchar
// ---
// function:
//   name: objectid
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: objectid_timestamp
//   section: temporal
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: timestamp(millisecond)
// ---
// function:
//   name: parse_data_size
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: decimal(38, 0)
// ---
// function:
//   name: parse_datetime
//   section: temporal
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: timestamp(millisecond)
// ---
// function:
//   name: parse_duration
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: interval day to second
// ---
// function:
//   name: parse_presto_data_size
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: decimal(38, 0)
// ---
// function:
//   name: percent_rank
//   section: other
//   kind: window
//   parameters: []
//   returns:
//     datatype: double
// ---
// function:
//   name: pi
//   section: other
//   kind: scalar
//   parameters: []
//   returns:
//     datatype: double
// ---
// function:
//   name: pow
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: double
//     - datatype: double
//   returns:
//     datatype: double
// ---
// function:
//   name: power
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: double
//     - datatype: double
//   returns:
//     datatype: double
// ---
// function:
//   name: qdigest_agg
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: bigint
//   returns:
//     datatype: varchar
// ---
// function:
//   name: qdigest_agg
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: bigint
//     - datatype: bigint
//   returns:
//     datatype: varchar
// ---
// function:
//   name: qdigest_agg
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: bigint
//     - datatype: bigint
//     - datatype: double
//   returns:
//     datatype: varchar
// ---
// function:
//   name: qdigest_agg
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: double
//   returns:
//     datatype: varchar
// ---
// function:
//   name: qdigest_agg
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: double
//     - datatype: bigint
//   returns:
//     datatype: varchar
// ---
// function:
//   name: qdigest_agg
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: double
//     - datatype: bigint
//     - datatype: double
//   returns:
//     datatype: varchar
// ---
// function:
//   name: qdigest_agg
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: float
//   returns:
//     datatype: varchar
// ---
// function:
//   name: qdigest_agg
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: float
//     - datatype: bigint
//   returns:
//     datatype: varchar
// ---
// function:
//   name: qdigest_agg
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: float
//     - datatype: bigint
//     - datatype: double
//   returns:
//     datatype: varchar
// ---
// function:
//   name: quantile_at_value
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: bigint
//   returns:
//     datatype: double
// ---
// function:
//   name: quantile_at_value
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: double
//   returns:
//     datatype: double
// ---
// function:
//   name: quantile_at_value
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: float
//   returns:
//     datatype: double
// ---
// function:
//   name: quarter
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: date
//   returns:
//     datatype: bigint
// ---
// function:
//   name: quarter
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: timestamp(p)
//   returns:
//     datatype: bigint
// ---
// function:
//   name: quarter
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: timestamp(p)
//   returns:
//     datatype: bigint
// ---
// function:
//   name: radians
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: double
//   returns:
//     datatype: double
// ---
// function:
//   name: rand
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: bigint
//   returns:
//     datatype: bigint
// ---
// function:
//   name: rand
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: bigint
//     - datatype: bigint
//   returns:
//     datatype: bigint
// ---
// function:
//   name: rand
//   section: other
//   kind: scalar
//   parameters: []
//   returns:
//     datatype: double
// ---
// function:
//   name: rand
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: int
//   returns:
//     datatype: int
// ---
// function:
//   name: rand
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: int
//     - datatype: int
//   returns:
//     datatype: int
// ---
// function:
//   name: rand
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: smallint
//   returns:
//     datatype: smallint
// ---
// function:
//   name: rand
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: smallint
//     - datatype: smallint
//   returns:
//     datatype: smallint
// ---
// function:
//   name: rand
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: tinyint
//   returns:
//     datatype: tinyint
// ---
// function:
//   name: rand
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: tinyint
//     - datatype: tinyint
//   returns:
//     datatype: tinyint
// ---
// function:
//   name: random
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: bigint
//   returns:
//     datatype: bigint
// ---
// function:
//   name: random
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: bigint
//     - datatype: bigint
//   returns:
//     datatype: bigint
// ---
// function:
//   name: random
//   section: other
//   kind: scalar
//   parameters: []
//   returns:
//     datatype: double
// ---
// function:
//   name: random
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: int
//   returns:
//     datatype: int
// ---
// function:
//   name: random
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: int
//     - datatype: int
//   returns:
//     datatype: int
// ---
// function:
//   name: random
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: smallint
//   returns:
//     datatype: smallint
// ---
// function:
//   name: random
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: smallint
//     - datatype: smallint
//   returns:
//     datatype: smallint
// ---
// function:
//   name: random
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: tinyint
//   returns:
//     datatype: tinyint
// ---
// function:
//   name: random
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: tinyint
//     - datatype: tinyint
//   returns:
//     datatype: tinyint
// ---
// function:
//   name: rank
//   section: other
//   kind: window
//   parameters: []
//   returns:
//     datatype: bigint
// ---
// function:
//   name: reduce
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: array(t)
//     - datatype: s
//     - datatype: lambda <(s, t) -> s>
//     - datatype: lambda <(s) -> r>
//   returns:
//     datatype: r
// ---
// function:
//   name: reduce_agg
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: t
//     - datatype: s
//     - datatype: lambda <(s, t) -> s>
//     - datatype: lambda <(s, s) -> s>
//   returns:
//     datatype: s
// ---
// function:
//   name: regexp_count
//   section: aggregate
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: bigint
// ---
// function:
//   name: regexp_extract
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: regexp_extract
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//     - datatype: bigint
//   returns:
//     datatype: varchar
// ---
// function:
//   name: regexp_extract_all
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: array<varchar>
// ---
// function:
//   name: regexp_extract_all
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//     - datatype: bigint
//   returns:
//     datatype: array<varchar>
// ---
// function:
//   name: regexp_like
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: boolean
// ---
// function:
//   name: regexp_position
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: int
// ---
// function:
//   name: regexp_position
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//     - datatype: bigint
//   returns:
//     datatype: int
// ---
// function:
//   name: regexp_position
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//     - datatype: bigint
//     - datatype: bigint
//   returns:
//     datatype: int
// ---
// function:
//   name: regexp_replace
//   section: string
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//     - datatype: lambda <(array<varchar>) -> varchar>
//   returns:
//     datatype: varchar
// ---
// function:
//   name: regexp_replace
//   section: string
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: regexp_replace
//   section: string
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: regexp_split
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: array<varchar>
// ---
// function:
//   name: regr_intercept
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: double
//     - datatype: double
//   returns:
//     datatype: double
// ---
// function:
//   name: regr_intercept
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: float
//     - datatype: float
//   returns:
//     datatype: float
// ---
// function:
//   name: regr_slope
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: double
//     - datatype: double
//   returns:
//     datatype: double
// ---
// function:
//   name: regr_slope
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: float
//     - datatype: float
//   returns:
//     datatype: float
// ---
// function:
//   name: regress
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: map<bigint, double>
//     - datatype: varchar
//   returns:
//     datatype: double
// ---
// function:
//   name: render
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: boolean
//   returns:
//     datatype: varchar
// ---
// function:
//   name: render
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: bigint
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: render
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: double
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: render
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: repeat
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: t
//     - datatype: bigint
//   returns:
//     datatype: array(t)
// ---
// function:
//   name: replace
//   section: string
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: replace
//   section: string
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: reverse
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: array(e)
//   returns:
//     datatype: array(e)
// ---
// function:
//   name: reverse
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: bytea
//   returns:
//     datatype: bytea
// ---
// function:
//   name: reverse
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: rgb
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: bigint
//     - datatype: bigint
//     - datatype: bigint
//   returns:
//     datatype: varchar
// ---
// function:
//   name: round
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: bigint
//   returns:
//     datatype: bigint
// ---
// function:
//   name: round
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: bigint
//     - datatype: bigint
//   returns:
//     datatype: bigint
// ---
// function:
//   name: round
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: decimal(p, s)
//   returns:
//     datatype: decimal(rp, rs)
// ---
// function:
//   name: round
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: decimal(p, s)
//     - datatype: bigint
//   returns:
//     datatype: decimal(rp, s)
// ---
// function:
//   name: round
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: double
//   returns:
//     datatype: double
// ---
// function:
//   name: round
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: double
//     - datatype: bigint
//   returns:
//     datatype: double
// ---
// function:
//   name: round
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: int
//   returns:
//     datatype: int
// ---
// function:
//   name: round
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: int
//     - datatype: bigint
//   returns:
//     datatype: int
// ---
// function:
//   name: round
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: float
//   returns:
//     datatype: float
// ---
// function:
//   name: round
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: float
//     - datatype: bigint
//   returns:
//     datatype: float
// ---
// function:
//   name: round
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: smallint
//   returns:
//     datatype: smallint
// ---
// function:
//   name: round
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: smallint
//     - datatype: bigint
//   returns:
//     datatype: smallint
// ---
// function:
//   name: round
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: tinyint
//   returns:
//     datatype: tinyint
// ---
// function:
//   name: round
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: tinyint
//     - datatype: bigint
//   returns:
//     datatype: tinyint
// ---
// function:
//   name: row_number
//   section: other
//   kind: window
//   parameters: []
//   returns:
//     datatype: bigint
// ---
// function:
//   name: rpad
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: bytea
//     - datatype: bigint
//     - datatype: bytea
//   returns:
//     datatype: bytea
// ---
// function:
//   name: rpad
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: bigint
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: rtrim
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: rtrim
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: rtrim
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: rtrim
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: second
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: interval day to second
//   returns:
//     datatype: bigint
// ---
// function:
//   name: second
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: time(p)
//   returns:
//     datatype: bigint
// ---
// function:
//   name: second
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: time(p)
//   returns:
//     datatype: bigint
// ---
// function:
//   name: second
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: timestamp(p)
//   returns:
//     datatype: bigint
// ---
// function:
//   name: second
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: timestamp(p)
//   returns:
//     datatype: bigint
// ---
// function:
//   name: sequence
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: bigint
//     - datatype: bigint
//   returns:
//     datatype: array<bigint>
// ---
// function:
//   name: sequence
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: bigint
//     - datatype: bigint
//     - datatype: bigint
//   returns:
//     datatype: array<bigint>
// ---
// function:
//   name: sequence
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: date
//     - datatype: date
//   returns:
//     datatype: array<date>
// ---
// function:
//   name: sequence
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: date
//     - datatype: date
//     - datatype: interval day to second
//   returns:
//     datatype: array<date>
// ---
// function:
//   name: sequence
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: date
//     - datatype: date
//     - datatype: interval day to second
//   returns:
//     datatype: array<date>
// ---
// function:
//   name: sequence
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: timestamp(p)
//     - datatype: timestamp(p)
//     - datatype: interval day to second
//   returns:
//     datatype: array(timestamp(p))
// ---
// function:
//   name: sequence
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: timestamp(p)
//     - datatype: timestamp(p)
//     - datatype: interval day to second
//   returns:
//     datatype: array(timestamp(p))
// ---
// function:
//   name: sha1
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: bytea
//   returns:
//     datatype: bytea
// ---
// function:
//   name: sha256
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: bytea
//   returns:
//     datatype: bytea
// ---
// function:
//   name: sha512
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: bytea
//   returns:
//     datatype: bytea
// ---
// function:
//   name: shuffle
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: array(e)
//   returns:
//     datatype: array(e)
// ---
// function:
//   name: sign
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: bigint
//   returns:
//     datatype: bigint
// ---
// function:
//   name: sign
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: decimal(p, s)
//   returns:
//     datatype: decimal(1, 0)
// ---
// function:
//   name: sign
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: double
//   returns:
//     datatype: double
// ---
// function:
//   name: sign
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: int
//   returns:
//     datatype: int
// ---
// function:
//   name: sign
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: float
//   returns:
//     datatype: float
// ---
// function:
//   name: sign
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: smallint
//   returns:
//     datatype: smallint
// ---
// function:
//   name: sign
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: tinyint
//   returns:
//     datatype: tinyint
// ---
// function:
//   name: simplify_geometry
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: double
//   returns:
//     datatype: varchar
// ---
// function:
//   name: sin
//   section: math
//   kind: scalar
//   parameters:
//     - datatype: double
//   returns:
//     datatype: double
// ---
// function:
//   name: sinh
//   section: math
//   kind: scalar
//   parameters:
//     - datatype: double
//   returns:
//     datatype: double
// ---
// function:
//   name: skewness
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: bigint
//   returns:
//     datatype: double
// ---
// function:
//   name: skewness
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: double
//   returns:
//     datatype: double
// ---
// function:
//   name: slice
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: array(e)
//     - datatype: bigint
//     - datatype: bigint
//   returns:
//     datatype: array(e)
// ---
// function:
//   name: soundex
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: spatial_partitioning
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: spatial_partitions
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: array<int>
// ---
// function:
//   name: spatial_partitions
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//     - datatype: double
//   returns:
//     datatype: array<int>
// ---
// function:
//   name: split
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: array<varchar>
// ---
// function:
//   name: split
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//     - datatype: bigint
//   returns:
//     datatype: array<varchar>
// ---
// function:
//   name: split_part
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//     - datatype: bigint
//   returns:
//     datatype: varchar
// ---
// function:
//   name: split_to_map
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: map<varchar, varchar>
// ---
// function:
//   name: split_to_multimap
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: map<varchar, array<varchar>>
// ---
// function:
//   name: spooky_hash_v2_32
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: bytea
//   returns:
//     datatype: bytea
// ---
// function:
//   name: spooky_hash_v2_64
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: bytea
//   returns:
//     datatype: bytea
// ---
// function:
//   name: sqrt
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: double
//   returns:
//     datatype: double
// ---
// function:
//   name: ST_Area
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: double
// ---
// function:
//   name: ST_Area
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: double
// ---
// function:
//   name: ST_AsBinary
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: bytea
// ---
// function:
//   name: ST_AsText
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: ST_Boundary
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: ST_Buffer
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: double
//   returns:
//     datatype: varchar
// ---
// function:
//   name: ST_Centroid
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: ST_Contains
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: boolean
// ---
// function:
//   name: ST_ConvexHull
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: ST_CoordDim
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: tinyint
// ---
// function:
//   name: ST_Crosses
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: boolean
// ---
// function:
//   name: ST_Difference
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: ST_Dimension
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: tinyint
// ---
// function:
//   name: ST_Disjoint
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: boolean
// ---
// function:
//   name: ST_Distance
//   section: math
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: double
// ---
// function:
//   name: ST_Distance
//   section: math
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: double
// ---
// function:
//   name: ST_EndPoint
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: ST_Envelope
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: ST_EnvelopeAsPts
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: array<varchar>
// ---
// function:
//   name: ST_Equals
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: boolean
// ---
// function:
//   name: ST_ExteriorRing
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: ST_Geometries
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: array<varchar>
// ---
// function:
//   name: ST_GeometryFromText
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: ST_GeometryN
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: bigint
//   returns:
//     datatype: varchar
// ---
// function:
//   name: ST_GeometryType
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: ST_GeomFromBinary
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: bytea
//   returns:
//     datatype: varchar
// ---
// function:
//   name: ST_InteriorRingN
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: bigint
//   returns:
//     datatype: varchar
// ---
// function:
//   name: ST_InteriorRings
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: array<varchar>
// ---
// function:
//   name: ST_Intersection
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: ST_Intersects
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: boolean
// ---
// function:
//   name: ST_IsClosed
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: boolean
// ---
// function:
//   name: ST_IsEmpty
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: boolean
// ---
// function:
//   name: ST_IsRing
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: boolean
// ---
// function:
//   name: ST_IsSimple
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: boolean
// ---
// function:
//   name: ST_IsValid
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: boolean
// ---
// function:
//   name: ST_Length
//   section: string
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: double
// ---
// function:
//   name: ST_Length
//   section: string
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: double
// ---
// function:
//   name: ST_LineFromText
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: ST_LineString
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: array<varchar>
//   returns:
//     datatype: varchar
// ---
// function:
//   name: ST_MultiPoint
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: array<varchar>
//   returns:
//     datatype: varchar
// ---
// function:
//   name: ST_NumGeometries
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: int
// ---
// function:
//   name: ST_NumInteriorRing
//   section: math
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: bigint
// ---
// function:
//   name: ST_NumPoints
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: bigint
// ---
// function:
//   name: ST_Overlaps
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: boolean
// ---
// function:
//   name: ST_Point
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: double
//     - datatype: double
//   returns:
//     datatype: varchar
// ---
// function:
//   name: ST_PointN
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: bigint
//   returns:
//     datatype: varchar
// ---
// function:
//   name: ST_Points
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: array<varchar>
// ---
// function:
//   name: ST_Polygon
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: ST_Relate
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: boolean
// ---
// function:
//   name: ST_StartPoint
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: ST_SymDifference
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: ST_Touches
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: boolean
// ---
// function:
//   name: ST_Union
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: ST_Within
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: boolean
// ---
// function:
//   name: ST_X
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: double
// ---
// function:
//   name: ST_XMax
//   section: math
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: double
// ---
// function:
//   name: ST_XMin
//   section: math
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: double
// ---
// function:
//   name: ST_Y
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: double
// ---
// function:
//   name: ST_YMax
//   section: math
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: double
// ---
// function:
//   name: ST_YMin
//   section: math
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: double
// ---
// function:
//   name: starts_with
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: boolean
// ---
// function:
//   name: stddev
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: bigint
//   returns:
//     datatype: double
// ---
// function:
//   name: stddev
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: double
//   returns:
//     datatype: double
// ---
// function:
//   name: stddev_pop
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: bigint
//   returns:
//     datatype: double
// ---
// function:
//   name: stddev_pop
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: double
//   returns:
//     datatype: double
// ---
// function:
//   name: stddev_samp
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: bigint
//   returns:
//     datatype: double
// ---
// function:
//   name: stddev_samp
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: double
//   returns:
//     datatype: double
// ---
// function:
//   name: strpos
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: bigint
// ---
// function:
//   name: strpos
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//     - datatype: bigint
//   returns:
//     datatype: bigint
// ---
// function:
//   name: substr
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: bytea
//     - datatype: bigint
//   returns:
//     datatype: bytea
// ---
// function:
//   name: substr
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: bytea
//     - datatype: bigint
//     - datatype: bigint
//   returns:
//     datatype: bytea
// ---
// function:
//   name: substr
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: bigint
//   returns:
//     datatype: varchar
// ---
// function:
//   name: substr
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: bigint
//     - datatype: bigint
//   returns:
//     datatype: varchar
// ---
// function:
//   name: substr
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: bigint
//   returns:
//     datatype: varchar
// ---
// function:
//   name: substr
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: bigint
//     - datatype: bigint
//   returns:
//     datatype: varchar
// ---
// function:
//   name: substring
//   section: string
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: bigint
//   returns:
//     datatype: varchar
// ---
// function:
//   name: substring
//   section: string
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: bigint
//     - datatype: bigint
//   returns:
//     datatype: varchar
// ---
// function:
//   name: substring
//   section: string
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: bigint
//   returns:
//     datatype: varchar
// ---
// function:
//   name: substring
//   section: string
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: bigint
//     - datatype: bigint
//   returns:
//     datatype: varchar
// ---
// function:
//   name: sum
//   section: math
//   kind: aggregate
//   parameters:
//     - datatype: bigint
//   returns:
//     datatype: bigint
// ---
// function:
//   name: sum
//   section: math
//   kind: aggregate
//   parameters:
//     - datatype: decimal(p, s)
//   returns:
//     datatype: decimal(38, s)
// ---
// function:
//   name: sum
//   section: math
//   kind: aggregate
//   parameters:
//     - datatype: double
//   returns:
//     datatype: double
// ---
// function:
//   name: sum
//   section: math
//   kind: aggregate
//   parameters:
//     - datatype: interval day to second
//   returns:
//     datatype: interval day to second
// ---
// function:
//   name: sum
//   section: math
//   kind: aggregate
//   parameters:
//     - datatype: interval day to second
//   returns:
//     datatype: interval day to second
// ---
// function:
//   name: sum
//   section: math
//   kind: aggregate
//   parameters:
//     - datatype: float
//   returns:
//     datatype: float
// ---
// function:
//   name: tan
//   section: math
//   kind: scalar
//   parameters:
//     - datatype: double
//   returns:
//     datatype: double
// ---
// function:
//   name: tanh
//   section: math
//   kind: scalar
//   parameters:
//     - datatype: double
//   returns:
//     datatype: double
// ---
// function:
//   name: tdigest_agg
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: double
//   returns:
//     datatype: varchar
// ---
// function:
//   name: tdigest_agg
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: double
//     - datatype: double
//   returns:
//     datatype: varchar
// ---
// function:
//   name: timestamp_objectid
//   section: temporal
//   kind: scalar
//   parameters:
//     - datatype: timestamp(second)
//   returns:
//     datatype: varchar
// ---
// function:
//   name: timezone_hour
//   section: temporal
//   kind: scalar
//   parameters:
//     - datatype: time(p)
//   returns:
//     datatype: bigint
// ---
// function:
//   name: timezone_hour
//   section: temporal
//   kind: scalar
//   parameters:
//     - datatype: timestamp(p)
//   returns:
//     datatype: bigint
// ---
// function:
//   name: timezone_minute
//   section: temporal
//   kind: scalar
//   parameters:
//     - datatype: time(p)
//   returns:
//     datatype: bigint
// ---
// function:
//   name: timezone_minute
//   section: math
//   kind: scalar
//   parameters:
//     - datatype: timestamp(p)
//   returns:
//     datatype: bigint
// ---
// function:
//   name: to_base
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: bigint
//     - datatype: bigint
//   returns:
//     datatype: varchar
// ---
// function:
//   name: to_base32
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: bytea
//   returns:
//     datatype: varchar
// ---
// function:
//   name: to_base64
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: bytea
//   returns:
//     datatype: varchar
// ---
// function:
//   name: to_base64url
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: bytea
//   returns:
//     datatype: varchar
// ---
// function:
//   name: to_big_endian_32
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: bigint
//   returns:
//     datatype: bytea
// ---
// function:
//   name: to_big_endian_64
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: bigint
//   returns:
//     datatype: bytea
// ---
// function:
//   name: to_char
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: timestamp(p)
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: to_date
//   section: temporal
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: date
// ---
// function:
//   name: to_encoded_polyline
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: to_geojson_geometry
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: to_geometry
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: to_hex
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: bytea
//   returns:
//     datatype: varchar
// ---
// function:
//   name: to_ieee754_32
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: float
//   returns:
//     datatype: bytea
// ---
// function:
//   name: to_ieee754_64
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: double
//   returns:
//     datatype: bytea
// ---
// function:
//   name: to_iso8601
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: date
//   returns:
//     datatype: varchar
// ---
// function:
//   name: to_iso8601
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: timestamp(p)
//   returns:
//     datatype: varchar
// ---
// function:
//   name: to_iso8601
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: timestamp(p)
//   returns:
//     datatype: varchar
// ---
// function:
//   name: to_milliseconds
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: interval day to second
//   returns:
//     datatype: bigint
// ---
// function:
//   name: to_spherical_geography
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: to_timestamp
//   section: temporal
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: timestamp(millisecond)
// ---
// function:
//   name: to_unixtime
//   section: temporal
//   kind: scalar
//   parameters:
//     - datatype: timestamp(p)
//   returns:
//     datatype: double
// ---
// function:
//   name: to_utf8
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: bytea
// ---
// function:
//   name: transform
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: array(t)
//     - datatype: lambda <(t) -> u>
//   returns:
//     datatype: array(u)
// ---
// function:
//   name: transform_keys
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: map(k1, v)
//     - datatype: lambda <(k1, v) -> k2>
//   returns:
//     datatype: map(k2, v)
// ---
// function:
//   name: transform_values
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: map(k, v1)
//     - datatype: lambda <(k, v1) -> v2>
//   returns:
//     datatype: map(k, v2)
// ---
// function:
//   name: translate
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: trim
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: trim
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: trim
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: trim
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: trim_array
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: array(e)
//     - datatype: bigint
//   returns:
//     datatype: array(e)
// ---
// function:
//   name: truncate
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: decimal(p, s)
//     - datatype: bigint
//   returns:
//     datatype: decimal(p, s)
// ---
// function:
//   name: truncate
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: decimal(p, s)
//   returns:
//     datatype: decimal(rp, 0)
// ---
// function:
//   name: truncate
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: double
//   returns:
//     datatype: double
// ---
// function:
//   name: truncate
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: float
//   returns:
//     datatype: float
// ---
// function:
//   name: typeof
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: t
//   returns:
//     datatype: varchar
// ---
// function:
//   name: upper
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: upper
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: url_decode
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: url_encode
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: url_extract_fragment
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: url_extract_host
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: url_extract_parameter
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: url_extract_path
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: url_extract_port
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: bigint
// ---
// function:
//   name: url_extract_protocol
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: url_extract_query
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: uuid
//   section: other
//   kind: scalar
//   parameters: []
//   returns:
//     datatype: varchar
// ---
// function:
//   name: value_at_quantile
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: double
//   returns:
//     datatype: double
// ---
// function:
//   name: values_at_quantiles
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: array<double>
//   returns:
//     datatype: array<double>
// ---
// function:
//   name: var_pop
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: bigint
//   returns:
//     datatype: double
// ---
// function:
//   name: var_pop
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: double
//   returns:
//     datatype: double
// ---
// function:
//   name: var_samp
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: bigint
//   returns:
//     datatype: double
// ---
// function:
//   name: var_samp
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: double
//   returns:
//     datatype: double
// ---
// function:
//   name: variance
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: bigint
//   returns:
//     datatype: double
// ---
// function:
//   name: variance
//   section: other
//   kind: aggregate
//   parameters:
//     - datatype: double
//   returns:
//     datatype: double
// ---
// function:
//   name: week
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: date
//   returns:
//     datatype: bigint
// ---
// function:
//   name: week
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: timestamp(p)
//   returns:
//     datatype: bigint
// ---
// function:
//   name: week
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: timestamp(p)
//   returns:
//     datatype: bigint
// ---
// function:
//   name: week_of_year
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: date
//   returns:
//     datatype: bigint
// ---
// function:
//   name: week_of_year
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: timestamp(p)
//   returns:
//     datatype: bigint
// ---
// function:
//   name: week_of_year
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: timestamp(p)
//   returns:
//     datatype: bigint
// ---
// function:
//   name: width_bucket
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: double
//     - datatype: array<double>
//   returns:
//     datatype: bigint
// ---
// function:
//   name: width_bucket
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: double
//     - datatype: double
//     - datatype: double
//     - datatype: bigint
//   returns:
//     datatype: bigint
// ---
// function:
//   name: wilson_interval_lower
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: bigint
//     - datatype: bigint
//     - datatype: double
//   returns:
//     datatype: double
// ---
// function:
//   name: wilson_interval_upper
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: bigint
//     - datatype: bigint
//     - datatype: double
//   returns:
//     datatype: double
// ---
// function:
//   name: with_timezone
//   section: temporal
//   kind: scalar
//   parameters:
//     - datatype: timestamp(p)
//     - datatype: varchar
//   returns:
//     datatype: timestamp(p)
// ---
// function:
//   name: word_stem
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: word_stem
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: varchar
// ---
// function:
//   name: xxhash64
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: bytea
//   returns:
//     datatype: bytea
// ---
// function:
//   name: year
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: date
//   returns:
//     datatype: bigint
// ---
// function:
//   name: year
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: interval day to second
//   returns:
//     datatype: bigint
// ---
// function:
//   name: year
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: timestamp(p)
//   returns:
//     datatype: bigint
// ---
// function:
//   name: year
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: timestamp(p)
//   returns:
//     datatype: bigint
// ---
// function:
//   name: year_of_week
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: date
//   returns:
//     datatype: bigint
// ---
// function:
//   name: year_of_week
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: timestamp(p)
//   returns:
//     datatype: bigint
// ---
// function:
//   name: year_of_week
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: timestamp(p)
//   returns:
//     datatype: bigint
// ---
// function:
//   name: yow
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: date
//   returns:
//     datatype: bigint
// ---
// function:
//   name: yow
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: timestamp(p)
//   returns:
//     datatype: bigint
// ---
// function:
//   name: yow
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: timestamp(p)
//   returns:
//     datatype: bigint
// ---
// function:
//   name: zip
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: array(t1)
//     - datatype: array(t2)
//   returns:
//     datatype: array(row(c0 t1, c1 t2))
// ---
// function:
//   name: zip
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: array(t1)
//     - datatype: array(t2)
//     - datatype: array(t3)
//   returns:
//     datatype: array(row(c0 t1, c1 t2, c2 t3))
// ---
// function:
//   name: zip
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: array(t1)
//     - datatype: array(t2)
//     - datatype: array(t3)
//     - datatype: array(t4)
//   returns:
//     datatype: array(row(c0 t1, c1 t2, c2 t3, c3 t4))
// ---
// function:
//   name: zip
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: array(t1)
//     - datatype: array(t2)
//     - datatype: array(t3)
//     - datatype: array(t4)
//     - datatype: array(t5)
//   returns:
//     datatype: array(row(c0 t1, c1 t2, c2 t3, c3 t4, c4 t5))
// ---
// function:
//   name: zip_with
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: array(t)
//     - datatype: array(u)
//     - datatype: lambda <(t, u) -> r>
//   returns:
//     datatype: array(r)
// ---
// function:
//   name: if
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: boolean
//     - datatype: T
//     - datatype: T
//   returns:
//     datatype: T
// ---
// function:
//   name: try
//   section: other
//   kind: scalar
//   parameters:
//     - datatype: T
//   returns:
//     datatype: T
// ---
// function:
//   name: format
//   section: other
//   variadic: uniform
//   kind: scalar
//   parameters:
//     - datatype: T
//   returns:
//     datatype: varchar
// ---
// function:
//   name: current_time
//   section: temporal
//   kind: scalar
//   parameters: []
//   returns:
//     datatype: time
// ---
// function:
//   name: current_timestamp
//   section: temporal
//   kind: scalar
//   parameters: []
//   returns:
//     datatype: timestamp(millisecond)
// ---
// function:
//   name: current_timestamp
//   section: temporal
//   kind: scalar
//   parameters:
//     - datatype: bigint
//       constant: "0"
//   returns:
//     datatype: timestamp(second)
// ---
// function:
//   name: current_timestamp
//   section: temporal
//   kind: scalar
//   parameters:
//     - datatype: bigint
//       constant: "3"
//   returns:
//     datatype: timestamp(millisecond)
// ---
// function:
//   name: current_timestamp
//   section: temporal
//   kind: scalar
//   parameters:
//     - datatype: bigint
//       constant: "6"
//   returns:
//     datatype: timestamp(microsecond)
// ---
// function:
//   name: current_timestamp
//   section: temporal
//   kind: scalar
//   parameters:
//     - datatype: bigint
//       constant: "9"
//   returns:
//     datatype: timestamp
// ---
// function:
//   name: localtime
//   section: temporal
//   kind: scalar
//   parameters: []
//   returns:
//     datatype: time
// ---
// function:
//   name: localtimestamp
//   section: temporal
//   kind: scalar
//   parameters: []
//   returns:
//     datatype: timestamp(millisecond)
// ---
// function:
//   name: localtimestamp
//   section: temporal
//   kind: scalar
//   parameters:
//     - datatype: bigint
//       constant: "0"
//   returns:
//     datatype: timestamp(second)
// ---
// function:
//   name: localtimestamp
//   section: temporal
//   kind: scalar
//   parameters:
//     - datatype: bigint
//       constant: "3"
//   returns:
//     datatype: timestamp(millisecond)
// ---
// function:
//   name: localtimestamp
//   section: temporal
//   kind: scalar
//   parameters:
//     - datatype: bigint
//       constant: "6"
//   returns:
//     datatype: timestamp(microsecond)
// ---
// function:
//   name: localtimestamp
//   section: temporal
//   kind: scalar
//   parameters:
//     - datatype: bigint
//       constant: "9"
//   returns:
//     datatype: timestamp
// ---
// function:
//   name: current_user
//   section: other
//   kind: scalar
//   parameters: []
//   returns:
//     datatype: varchar
// ---
// function:
//   name: current_catalog
//   section: other
//   kind: scalar
//   parameters: []
//   returns:
//     datatype: varchar
// ---
// function:
//   name: current_schema
//   section: other
//   kind: scalar
//   parameters: []
//   returns:
//     datatype: varchar
// ---
// function:
//   name: reclassify
//   kind: scalar
//   parameters:
//     - datatype: T
//     - datatype: varchar
//     - datatype: varchar
//   returns:
//     datatype: T
//   description:
//     "Changes the classification label of the first argument from the expected
//     classifier specified by the second argument to the desired classifier
//     specified by the third. The expected and desired classifiers
//     must belong to the same classifier group. No other classifiers attached to
//     the first argument are impacted. The function will emit a warning if the
//     the first argument does not have the expected classifier"
//   examples:
//     - input: select reclassify(12345, 'pii.clear_text', 'pii.masked') as value;
//       output: "12345"
//   section: other
// ---
// function:
//   name: reclassify
//   kind: scalar
//   parameters:
//     - datatype: T
//     - datatype: varchar
//   returns:
//     datatype: T
//   description: "Changes the classification label of the first argument
//     to the desired classifier specified by the second argument. The first
//     argument is expected to have one or more classifiers from the same
//     classifier group as the desired classifier. All of these classifiers
//     will be removed and replaced by the desired classifier. No other classifiers
//     attached to the first argument will be impacted. The function will emit a warning
//     if the first argument does not have a classifier from the expected classifier group"
//   examples:
//     - input: select reclassify(12345, 'pii.masked') as value;
//       output: "12345"
//   section: other

// Function package declaration
pub struct TestFunctionPackage;

impl ScalarFunctionPackage for TestFunctionPackage {
    fn functions(&self) -> Vec<Box<dyn ScalarFunctionDef>> {
        vec![
            Box::new(AddOneFunction),
            Box::new(MultiplyTwoFunction),
            Box::new(ArrayDistinctFunction),
        ]
    }
}

#[cfg(test)]
mod test {
    use arrow::{
        array::ArrayRef, record_batch::RecordBatch, util::display::array_value_to_string,
    };
    use datafusion::error::Result;
    use datafusion::prelude::SessionContext;
    use tokio;

    use crate::TestFunctionPackage;

    /// Execute query and return result set as 2-d table of Vecs
    /// `result[row][column]`
    async fn execute(ctx: &SessionContext, sql: &str) -> Vec<Vec<String>> {
        result_vec(&execute_to_batches(ctx, sql).await)
    }

    /// Specialised String representation
    fn col_str(column: &ArrayRef, row_index: usize) -> String {
        if column.is_null(row_index) {
            return "NULL".to_string();
        }

        array_value_to_string(column, row_index)
            .ok()
            .unwrap_or_else(|| "???".to_string())
    }
    /// Converts the results into a 2d array of strings, `result[row][column]`
    /// Special cases nulls to NULL for testing
    fn result_vec(results: &[RecordBatch]) -> Vec<Vec<String>> {
        let mut result = vec![];
        for batch in results {
            for row_index in 0..batch.num_rows() {
                let row_vec = batch
                    .columns()
                    .iter()
                    .map(|column| col_str(column, row_index))
                    .collect();
                result.push(row_vec);
            }
        }
        result
    }

    /// Execute query and return results as a Vec of RecordBatches
    async fn execute_to_batches(ctx: &SessionContext, sql: &str) -> Vec<RecordBatch> {
        let df = ctx.sql(sql).await.unwrap();

        // optimize just for check schema don't change during optimization.
        df.clone().into_optimized_plan().unwrap();

        df.collect().await.unwrap()
    }

    macro_rules! test_expression {
        ($SQL:expr, $EXPECTED:expr) => {
            let ctx = SessionContext::new();
            ctx.register_scalar_function_package(Box::new(TestFunctionPackage));
            let sql = format!("SELECT {}", $SQL);
            let actual = execute(&ctx, sql.as_str()).await;
            assert_eq!(actual[0][0], $EXPECTED);
        };
    }

    #[tokio::test]
    async fn test_add_one() -> Result<()> {
        test_expression!("add_one(1)", "2.0");
        test_expression!("add_one(-1)", "0.0");
        Ok(())
    }

    #[tokio::test]
    async fn test_multiply_two() -> Result<()> {
        test_expression!("multiply_two(1)", "2.0");
        test_expression!("multiply_two(-1)", "-2.0");
        Ok(())
    }

    #[tokio::test]
    async fn test_array_distinct() -> Result<()> {
        test_expression!("array_distinct([1,2,3,2,1])", "[1,2,3]");
        Ok(())
    }
}
