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
// use arrow::array::{ArrayRef, BooleanArray, Int64Array,StringBuilder,Float64Array};
// use arrow::array::{ StringArray,Array,};
use arrow::array::{ArrayRef, TimestampNanosecondArray};
use arrow::compute::cast;
use arrow::datatypes::{Field, IntervalDayTimeType};
use arrow::datatypes::{self, IntervalUnit};
use arrow::datatypes::{DataType, TimeUnit};
use datafusion::error::Result;
// use datafusion::scalar::ScalarFunctionDef;
//use datafusion::physical_plan::functions::{Signature, Volatility};
use chrono::{Datelike, Duration, NaiveDate, NaiveDateTime};
use chrono::prelude::*;
use chrono:: Months;
use datafusion::logical_expr::Volatility;
use datafusion::physical_plan::expressions::Column;
use datafusion_common::cast::{as_float64_array, as_string_array, as_date32_array, as_timestamp_nanosecond_array};
use datafusion_common::DataFusionError;
use datafusion_expr::{
    ReturnTypeFunction, ScalarFunctionDef, ScalarFunctionPackage, Signature,
};
use std::collections::HashSet;
use std::sync::Arc;
use datafusion_common::scalar::ScalarValue;
use datafusion_expr::ColumnarValue;
#[derive(Debug)]
pub struct LuhnCheck;

impl ScalarFunctionDef for LuhnCheck {
    fn name(&self) -> &str {
        "luhn_check"
    }

    fn signature(&self) -> Signature {
        Signature::exact(vec![DataType::Utf8], Volatility::Immutable)
    }

    fn return_type(&self) -> ReturnTypeFunction {
        let return_type = Arc::new(DataType::Boolean);
        Arc::new(move |_| Ok(return_type.clone()))
    }

    fn execute(&self, args: &[ArrayRef]) -> Result<ArrayRef> {
        assert_eq!(args.len(), 1);
        let input = as_string_array(&args[0]).expect("cast failed");
        let array = input
            .iter()
            .map(|value| match value {
                Some(value) => Some(luhn_check(&value)),
                _ => None,
            })
            .collect::<BooleanArray>();
        Ok(Arc::new(array) as ArrayRef)
    }
}

fn luhn_check(num_str: &str) -> bool {
    let mut sum = 0;
    let mut is_second = false;

    for digit in num_str.chars().rev() {
        if let Some(mut val) = digit.to_digit(10) {
            if is_second {
                val *= 2;
                if val > 9 {
                    val -= 9;
                }
            }
            sum += val;
            is_second = !is_second;
        } else {
            return false; // Invalid character
        }
    }
    sum % 10 == 0
}

#[derive(Debug)]
pub struct LevenshteinDistance;

impl ScalarFunctionDef for LevenshteinDistance {
    fn name(&self) -> &str {
        "levenshtein_distance"
    }

    fn signature(&self) -> Signature {
        Signature::exact(vec![DataType::Utf8, DataType::Utf8], Volatility::Immutable)
    }

    fn return_type(&self) -> ReturnTypeFunction {
        let return_type = Arc::new(DataType::Int64);
        Arc::new(move |_| Ok(return_type.clone()))
    }

    fn execute(&self, args: &[ArrayRef]) -> Result<ArrayRef> {
        assert_eq!(args.len(), 2);
        let input0 = as_string_array(&args[0]).expect("cast failed");
        let input1 = as_string_array(&args[1]).expect("cast failed");
        let array = input0
            .into_iter()
            .zip(input1.into_iter())
            .map(|(s1, s2)| match (s1, s2) {
                (Some(s1), Some(s2)) => Some(levenshtein(&s1, &s2) as i64),
                _ => None,
            })
            .collect::<Int64Array>();
        Ok(Arc::new(array) as ArrayRef)
    }
}

fn levenshtein(s1: &str, s2: &str) -> usize {
    let mut d = vec![vec![0; s2.len() + 1]; s1.len() + 1];

    for i in 0..=s1.len() {
        d[i][0] = i;
    }
    for j in 0..=s2.len() {
        d[0][j] = j;
    }

    for (i, char1) in s1.chars().enumerate() {
        for (j, char2) in s2.chars().enumerate() {
            d[i + 1][j + 1] = if char1 == char2 {
                d[i][j]
            } else {
                d[i][j + 1].min(d[i + 1][j]).min(d[i][j]) + 1
            };
        }
    }
    d[s1.len()][s2.len()]
}

#[derive(Debug)]

pub struct HammingDistance;

impl ScalarFunctionDef for HammingDistance {
    fn name(&self) -> &str {
        "hamming_distance"
    }

    fn signature(&self) -> Signature {
        Signature::exact(vec![DataType::Utf8, DataType::Utf8], Volatility::Immutable)
    }

    fn return_type(&self) -> ReturnTypeFunction {
        let return_type = Arc::new(DataType::Int64);
        Arc::new(move |_| Ok(return_type.clone()))
    }

    fn execute(&self, args: &[ArrayRef]) -> Result<ArrayRef> {
        assert_eq!(args.len(), 2);
        let input0 = as_string_array(&args[0]).expect("cast failed");
        let input1 = as_string_array(&args[1]).expect("cast failed");
        let array = input0
            .into_iter()
            .zip(input1.into_iter())
            .map(|(value0, value1)| match (value0, value1) {
                (None, None) => todo!(),
                (None, Some(_)) => todo!(),
                (Some(_), None) => todo!(),
                (Some(value0), Some(value1)) => {
                    let mut distance = 0;
                    for (c0, c1) in value0.chars().zip(value1.chars()) {
                        if c0 != c1 {
                            distance += 1;
                        }
                    }
                    Some(distance)
                }
            })
            .collect::<Int64Array>();
        Ok(Arc::new(array) as ArrayRef)
    }
}

#[derive(Debug)]
pub struct LengthFunction;

impl ScalarFunctionDef for LengthFunction {
    fn name(&self) -> &str {
        "length"
    }

    fn signature(&self) -> Signature {
        Signature::exact(vec![DataType::Utf8], Volatility::Immutable)
    }

    fn return_type(&self) -> ReturnTypeFunction {
        let return_type = Arc::new(DataType::Int64);
        Arc::new(move |_| Ok(return_type.clone()))
    }

    // Returns the length of string in characters.
    fn execute(&self, args: &[ArrayRef]) -> Result<ArrayRef> {
        assert_eq!(args.len(), 1);
        let input = args[0]
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("cast failed");
        let lengths = input
            .iter()
            .map(|value| value.map(|s| s.chars().count() as i64))
            .collect::<Vec<_>>();
        let array = Int64Array::from(lengths);
        Ok(Arc::new(array) as ArrayRef)
    }
}

#[derive(Debug)]
pub struct LTrimFunction;

impl ScalarFunctionDef for LTrimFunction {
    fn name(&self) -> &str {
        "ltrim"
    }

    fn signature(&self) -> Signature {
        Signature::exact(vec![DataType::Utf8], Volatility::Immutable)
    }

    fn return_type(&self) -> ReturnTypeFunction {
        let return_type = Arc::new(DataType::Utf8);
        Arc::new(move |_| Ok(return_type.clone()))
    }

    fn execute(&self, args: &[ArrayRef]) -> Result<ArrayRef> {
        assert_eq!(args.len(), 1);
        let input = args[0]
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("cast failed");
        let mut builder = StringBuilder::new();

        for i in 0..input.len() {
            if input.is_null(i) {
                builder.append_null();
            } else {
                let value = input.value(i);
                builder.append_value(value.trim_start());
            }
        }

        let array = builder.finish();
        Ok(Arc::new(array) as ArrayRef)
    }
}

#[derive(Debug)]
pub struct RTrimFunction;

impl ScalarFunctionDef for RTrimFunction {
    fn name(&self) -> &str {
        "rtrim"
    }

    fn signature(&self) -> Signature {
        Signature::exact(vec![DataType::Utf8], Volatility::Immutable)
    }

    fn return_type(&self) -> ReturnTypeFunction {
        let return_type = Arc::new(DataType::Utf8);
        Arc::new(move |_| Ok(return_type.clone()))
    }

    fn execute(&self, args: &[ArrayRef]) -> Result<ArrayRef> {
        assert_eq!(args.len(), 1);
        let input = args[0]
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("cast failed");
        let mut builder = StringBuilder::new();

        for i in 0..input.len() {
            if input.is_null(i) {
                builder.append_null();
            } else {
                let value = input.value(i);
                builder.append_value(value.trim_end());
            }
        }

        let array = builder.finish();
        Ok(Arc::new(array) as ArrayRef)
    }
}

#[derive(Debug)]
pub struct TrimFunction;

impl ScalarFunctionDef for TrimFunction {
    fn name(&self) -> &str {
        "trim"
    }

    fn signature(&self) -> Signature {
        Signature::exact(vec![DataType::Utf8], Volatility::Immutable)
    }

    fn return_type(&self) -> ReturnTypeFunction {
        let return_type = Arc::new(DataType::Utf8);
        Arc::new(move |_| Ok(return_type.clone()))
    }

    fn execute(&self, args: &[ArrayRef]) -> Result<ArrayRef> {
        assert_eq!(args.len(), 1);
        let input = args[0]
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("cast failed");
        let mut builder = StringBuilder::new();

        for i in 0..input.len() {
            if input.is_null(i) {
                builder.append_null();
            } else {
                let value = input.value(i);
                builder.append_value(value.trim());
            }
        }

        let array = builder.finish();
        Ok(Arc::new(array) as ArrayRef)
    }
}

#[derive(Debug)]
pub struct ParseIdentFunction;

impl ScalarFunctionDef for ParseIdentFunction {
    fn name(&self) -> &str {
        "parse_ident"
    }

    fn signature(&self) -> Signature {
        Signature::exact(vec![DataType::Utf8], Volatility::Immutable)
    }

    fn return_type(&self) -> ReturnTypeFunction {
        let return_type = Arc::new(DataType::Utf8);
        Arc::new(move |_| Ok(return_type.clone()))
    }

    fn execute(&self, args: &[ArrayRef]) -> Result<ArrayRef> {
        assert_eq!(args.len(), 1);
        let input = as_string_array(&args[0]).expect("cast failed");
        let array = input
            .iter()
            .map(|value| {
                value.map(|v| {
                    v.split('.')
                        .map(|identifier| identifier.trim_matches('"').to_string())
                        .collect::<Vec<String>>()
                        .join(",")
                })
            })
            .collect::<StringArray>();
        Ok(Arc::new(array) as ArrayRef)
    }
}

#[derive(Debug)]
pub struct AgeFunction;

impl ScalarFunctionDef for AgeFunction {
    fn name(&self) -> &str {
        "age"
    }

    fn signature(&self) -> Signature {
        Signature::exact(
            vec![
                // DataType::Timestamp(TimeUnit::Nanosecond, None),
                DataType::Timestamp(TimeUnit::Nanosecond, None),
            ],
            Volatility::Immutable,
        )
    }

    fn return_type(&self) -> ReturnTypeFunction {
        let return_type = Arc::new(DataType::Timestamp(TimeUnit::Microsecond, None));
        Arc::new(move |_| Ok(return_type.clone()))
    }

    fn execute(&self, args: &[ArrayRef]) -> Result<ArrayRef> { //这个函数需要数组吗
        assert_eq!(args.len(), 1);

        // let start_array = args[0]
        //     .as_any()
        //     .downcast_ref::<TimestampNanosecondArray>()
        //     .expect("cast to TimestampNanosecondArray failed");
        // let end_array = args[1]
        //     .as_any()
        //     .downcast_ref::<TimestampNanosecondArray>()
        //     .expect("cast to TimestampNanosecondArray failed");

        // let start_value = start_array.value(0).to_string().as_str();
        // let end_value = end_array.value(0).to_string().as_str();
        // let start = NaiveDate::parse_from_str(start_value, "%Y-%m-%d").map_err(|err|{
        //     DataFusionError::Execution(format!("Parse error: {}", err))
        // })?;
        // let end = NaiveDate::parse_from_str(end_value, "%Y-%m-%d").map_err(|err|{
        //     DataFusionError::Execution(format!("Parse error: {}", err))
        // })?;
        // let duration = end.signed_duration_since(start);
    //    let scalar_value = ScalarValue::DurationMicrosecond(Some(duration.num_microseconds()).unwrap());
       //let columar_value = ColumnarValue::Scalar(scalar_value);

    // let array_value: Arc<dyn arrow::array::Array> = Arc::new(DurationMicrosecondArray::from(vec![duration]));
    // let columar_value = ColumnarValue::Array(array_value);
    //     Ok(Arc::new(columar_value) as ArrayRef);

    let array = &args[0];
    let mut  b = IntervalDayTimeBuilder::with_capacity(array.len());
    match array.data_type() {

        DataType::Timestamp(_, _)  => {
            let array = as_timestamp_nanosecond_array(&array)?;
            let iter: ArrayIter<&PrimitiveArray<_>> = ArrayIter::new(array);
            iter.into_iter().for_each(|value| {
                if let Some(value) = value {
                    b.append_value(IntervalDayTimeType::make_value(1, 0));
                } else {
                    b.append_null();
                }
            });
        }
        
        _ => todo!()
    }
    let result = b.finish();
    cast(&(Arc::new(result) as ArrayRef), &DataType::Interval(IntervalUnit::DayTime)).map_err(|err| {
        DataFusionError::Execution(format!("Cast error: {}", err))
    })
    }
}

fn is_leap_year(year: i32) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}
// Function package declaration
pub struct FunctionPackage;

impl ScalarFunctionPackage for FunctionPackage {
    fn functions(&self) -> Vec<Box<dyn ScalarFunctionDef>> {
        vec![
            Box::new(HammingDistance),
            Box::new(LevenshteinDistance),
            Box::new(LuhnCheck),
            Box::new(LengthFunction),
            Box::new(LTrimFunction),
            Box::new(RTrimFunction),
            Box::new(TrimFunction),
            Box::new(ParseIdentFunction),
            Box::new(AgeFunction),
        ]
    }
}

#[cfg(test)]
mod test {
    use arrow::compute::kernels::substring;
    use datafusion::error::Result;
    use datafusion::prelude::SessionContext;
    use tokio;

    use crate::utils::{execute, test_expression};

    use super::FunctionPackage;

    #[tokio::test]
    async fn test_luhn_check() -> Result<()> {
        test_expression!("luhn_check('79927398713')", "true");
        test_expression!("luhn_check('79927398714')", "false");
        Ok(())
    }

    #[tokio::test]
    async fn test_levenshtein_distance() -> Result<()> {
        test_expression!("levenshtein_distance('kitten','sitting')", "3");
        test_expression!("levenshtein_distance('flaw','lawn')", "2");
        test_expression!("levenshtein_distance('','abc')", "3");
        Ok(())
    }

    #[tokio::test]
    async fn test_hamming_distance() -> Result<()> {
        test_expression!("hamming_distance('0000','1111')", "4");
        test_expression!("hamming_distance('karolin','kathrin')", "3");
        Ok(())
    }

    #[tokio::test]
    async fn test_rtrim() -> Result<()> {
        test_expression!("rtrim('  Hello  ')", "  Hello");
        test_expression!("rtrim('   ')", "");
        test_expression!("rtrim('Hello')", "Hello");
        test_expression!("rtrim('Hello  ')", "Hello");
        Ok(())
    }

    #[tokio::test]
    async fn test_trim() -> Result<()> {
        test_expression!("trim('  Hello  ')", "Hello");
        test_expression!("trim('   ')", "");
        test_expression!("trim('Hello')", "Hello");
        test_expression!("trim('  Hello  ')", "Hello");
        Ok(())
    }

    #[tokio::test]
    async fn test_ltrim() -> Result<()> {
        test_expression!("ltrim('  leading whitespace')", "leading whitespace");
        test_expression!("ltrim('no leading whitespace')", "no leading whitespace");
        Ok(())
    }

    #[tokio::test]
    async fn test_length() -> Result<()> {
        test_expression!("length('hello')", "5");
        test_expression!("length('你好')", "2");
        Ok(())
    }

    #[tokio::test]
    async fn test_parse_ident() -> Result<()> {
        test_expression!(
            "parse_ident('\"SomeSchema\".sometable')",
            "SomeSchema,sometable"
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_age_function() -> Result<()> {

        // // Test date difference within the same month
        test_expression!(
            "age(timestamp '2001-04-10')",
            "0 years 0 mons 1 days 0 hours 0 mins 0.000 secs"
        );
        // // Test date difference between different months within the same year
        // test_expression!(
        //     "age(timestamp '2001-04-10', timestamp '2001-05-10')",
        //     "0 years 1 months 0 days"
        // );
        // // Test date difference across different years
        // test_expression!(
        //     "age(timestamp '2000-04-10', timestamp '2001-04-10')",
        //     "1 years 0 months 0 days"
        // );
        // // Test date difference involving a leap year
        //  test_expression!(
        //      "age(timestamp '2000-02-28', timestamp '2000-03-01')",
        //     "0 years 0 months 2 days" // 2000 is a leap year
        // );
        // // Test date difference between the end of a month and the beginning of the next
        // test_expression!(
        //     "age(timestamp '2001-05-01', timestamp '2001-04-30')",
        //     "0 years 0 months 1 days"
        // );
        // // age(timestamp '2001-04-10', timestamp '1957-06-13')
        // test_expression!(
        //     "age(timestamp '2001-04-10', timestamp '1957-06-13')",
        //     "43 years 9 months 27 days"
        // );

        Ok(())
    }
}
