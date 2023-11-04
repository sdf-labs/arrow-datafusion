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
use arrow::datatypes;
use arrow::datatypes::DataType;
use datafusion::error::Result;
use datafusion::logical_expr::Volatility;
use datafusion_common::cast::as_int32_array;
use datafusion_common::cast::as_int64_array;
use datafusion_common::cast::*;
use datafusion_common::cast::{as_float64_array, as_string_array};
use datafusion_common::DataFusionError;
use datafusion_expr::{
    ReturnTypeFunction, ScalarFunctionDef, ScalarFunctionPackage, Signature,
};
use std::collections::HashSet;
use std::sync::Arc;

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
            .map(|value| value.map(|s| s.len() as i64))
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

// #[derive(Debug)]
// pub struct SplitPartFunction;

// impl ScalarFunctionDef for SplitPartFunction {
//     fn name(&self) -> &str {
//         "split_part"
//     }

//     fn signature(&self) -> Signature {
//         Signature::exact(
//             vec![DataType::Utf8, DataType::Utf8, DataType::Int32],
//             Volatility::Immutable,
//         )
//     }

//     fn return_type(&self) -> ReturnTypeFunction {
//         let return_type = Arc::new(DataType::Utf8);
//         Arc::new(move |_| Ok(return_type.clone()))
//     }

//     fn execute(&self, args: &[ArrayRef]) -> Result<ArrayRef> {
//         assert_eq!(args.len(), 3);

//         let string = as_string_array(&args[0])?;
//         let delimiter = as_string_array(&args[1])?;
//         let position = as_int64_array(&args[2])?;

//         let mut results = Vec::with_capacity(string.len());

//         for i in 0..string.len() {
//             let string_value = string.value(i);
//             let delimiter_value = delimiter.value(i);
//             let mut parts = string_value.split(delimiter_value);
//             let pos = position.value(i) as usize;

//             let parts = string
//                 .value(i)
//                 .split(delimiter.value(i))
//                 .collect::<Vec<_>>();

//             let index = if pos < 0 {
//                 // Convert negative index to positive index
//                 (parts.len() as i64 + pos as i64) as usize
//             } else {
//                 pos as usize
//             };

//             let part = parts.get(index).map(|s| s.to_string()).unwrap_or_default();
//             results.push(part);
//         }

//         let array = StringArray::from(results);
//         Ok(Arc::new(array) as ArrayRef)
//     }
// }

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
            //Box::new(SplitPartFunction),
            Box::new(ParseIdentFunction),
        ]
    }
}

#[cfg(test)]
mod test {
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
    async fn test_split_part() -> Result<()> {
        test_expression!("split_part('abc~@~def~@~ghi', '~@~', 2)", "def");
        test_expression!("split_part('abc,def,ghi,jkl', ',', -2)", "ghi");
        // zip 数组分割 在examples上找
        Ok(())
    }

    #[tokio::test]
    async fn test_parse_ident() -> Result<()> {
        test_expression!("parse_ident('\"SomeSchema\".sometable')", "SomeSchema,sometable");
        // Add other test cases if needed
        Ok(())
    }
    
}

// #[cfg(test)]
//     mod tests {
//         use crate::presto::ParseIdentFunction;
//         use arrow::array::{ArrayRef, StringArray};
//         use crate::presto::ScalarFunctionDef;// text引用struct也要导入相应的impl
//         use std::sync::Arc;

//         #[test]
//         fn test_parse_ident() {
//             // 创建一个示例输入：一个包含单个字符串的 StringArray
//             let input_array = StringArray::from(vec!['"SomeSchema".someTable']);
//             let input_arc = Arc::new(input_array) as ArrayRef;

//             // 创建并执行 ParseIdentFunction
//             let func = ParseIdentFunction;
//             let result = func.execute(&[input_arc]).unwrap();

            
//             // 将结果转换回 StringArray 并提取第一个（且唯一的）值
//             let result_array = result.as_any().downcast_ref::<StringArray>().unwrap();
//             let result_value = result_array.value(0);

//             // 验证结果是否与预期相符
//             assert_eq!(result_value, "SomeSchema,sometable");
//         }
//     }
