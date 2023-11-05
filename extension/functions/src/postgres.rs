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

use arrow::array::{ArrayRef, Int64Array, BooleanArray, StringArray};
use arrow::datatypes::DataType;
use datafusion::error::Result;
use datafusion::logical_expr::Volatility;
use datafusion_common::cast::{as_string_array, as_int64_array};
use datafusion_expr::{
    ReturnTypeFunction, ScalarFunctionDef, ScalarFunctionPackage, Signature, TypeSignature,
};
use std::sync::Arc;

use regex::Regex;

#[derive(Debug)]
pub struct RegexpCountFunction;

impl ScalarFunctionDef for RegexpCountFunction {
    fn name(&self) -> &str {
        "regexp_count"
    }

    fn signature(&self) -> Signature {
        Signature::one_of(
            vec![
                TypeSignature::Exact(vec![DataType::Utf8, DataType::Utf8]),
                TypeSignature::Exact(vec![DataType::Utf8, DataType::Utf8, DataType::Int64]),
                TypeSignature::Exact(vec![DataType::Utf8, DataType::Utf8, DataType::Int64, DataType::Utf8])
            ],
            Volatility::Immutable
        )        
    }
    

    fn return_type(&self) -> ReturnTypeFunction {
        let return_type = Arc::new(DataType::Int64);
        Arc::new(move |_| Ok(return_type.clone()))
    }

    fn execute(&self, args: &[ArrayRef]) -> Result<ArrayRef> {
        // 确保至少有2个参数：input 和 pattern
        assert!(args.len() >= 2);
    
        // 从参数中获取 input 和 pattern
        let input = as_string_array(&args[0]).expect("cast failed");
        let pattern = as_string_array(&args[1]).expect("cast failed");
    
        // 检查是否提供了 start 参数
        let start = if args.len() >= 3 {
            as_int64_array(&args[2]).expect("cast failed").iter().next().unwrap_or(None)
        } else {
            None
        };
        // 检查是否提供了 flag 参数
        let flag_string = if args.len() >= 4 {
            let flags_array = as_string_array(&args[3]).expect("cast failed");
            if let Some(flag) = flags_array.iter().next().unwrap_or(None) {
                format!("(?{})", flag)
            } else {
                String::new()
            }
        } else {
            String::new()
        };        
    
        // 进行正则表达式匹配
        let array = input.into_iter().zip(pattern.into_iter()).map(|(text, pat)| {
            if let (Some(text), Some(pat)) = (text, pat) {
                // 如果提供了 start 参数，则从指定位置开始搜索
                let text = &text[start.unwrap_or(0) as usize..];
                // 创建正则表达式
                let re = Regex::new(&format!("{}{}", flag_string, pat)).expect("Invalid regex pattern");
                // 计算匹配次数
                Some(re.find_iter(text).count() as i64)
            } else {
                None
            }
        }).collect::<Int64Array>();
        
        // 返回结果
        Ok(Arc::new(array) as ArrayRef)
    }    
}

#[derive(Debug)]
pub struct RegexpLikeFunction;

impl ScalarFunctionDef for RegexpLikeFunction {
    fn name(&self) -> &str {
        "regexp_like"
    }

    fn signature(&self) -> Signature {
        Signature::one_of(
            vec![
                TypeSignature::Exact(vec![DataType::Utf8, DataType::Utf8]),
                TypeSignature::Exact(vec![DataType::Utf8, DataType::Utf8, DataType::Utf8]),
            ],
            Volatility::Immutable
        )
    }

    fn return_type(&self) -> ReturnTypeFunction {
        let return_type = Arc::new(DataType::Boolean);
        Arc::new(move |_| Ok(return_type.clone()))
    }

    fn execute(&self, args: &[ArrayRef]) -> Result<ArrayRef> {
        assert!(args.len() >= 2);

        let input = as_string_array(&args[0]).expect("cast failed");
        let pattern = as_string_array(&args[1]).expect("cast failed");

        let flag_string = if args.len() >= 3 {
            let flags_array = as_string_array(&args[2]).expect("cast failed");
            if let Some(flag) = flags_array.iter().next().unwrap_or(None) {
                format!("(?{})", flag)
            } else {
                String::new()
            }
        } else {
            String::new()
        };        

        let array = input.into_iter().zip(pattern.into_iter()).map(|(text, pat)| {
            if let (Some(text), Some(pat)) = (text, pat) {
                let re = Regex::new(&format!("{}{}", flag_string, pat)).expect("Invalid regex pattern");
                Some(re.is_match(text))
            } else {
                None
            }
        }).collect::<BooleanArray>();

        Ok(Arc::new(array) as ArrayRef)
    }    
}

#[derive(Debug)]
pub struct RegexpReplaceFunction;

impl ScalarFunctionDef for RegexpReplaceFunction {
    fn name(&self) -> &str {
        "regexp_replace"
    }

    fn signature(&self) -> Signature {
        Signature::one_of(
            vec![
                TypeSignature::Exact(vec![DataType::Utf8, DataType::Utf8, DataType::Utf8, DataType::Int64, DataType::Int64]),
                TypeSignature::Exact(vec![DataType::Utf8, DataType::Utf8, DataType::Utf8, DataType::Int64, DataType::Int64, DataType::Utf8])
            ],
            Volatility::Immutable
        )        
    }
    
    fn return_type(&self) -> ReturnTypeFunction {
        let return_type = Arc::new(DataType::Utf8);
        Arc::new(move |_| Ok(return_type.clone()))
    }

    fn execute(&self, args: &[ArrayRef]) -> Result<ArrayRef> {
        assert!(args.len() >= 5);
    
        let input = as_string_array(&args[0]).expect("cast failed");
        let pattern = as_string_array(&args[1]).expect("cast failed");
        let replacement = as_string_array(&args[2]).expect("cast failed");
        let start = as_int64_array(&args[3]).expect("cast failed").iter().next().unwrap_or(None);
        let n = as_int64_array(&args[4]).expect("cast failed").iter().next().unwrap_or(None);
    
        let flag_string = if args.len() == 6 {
            let flags_array = as_string_array(&args[5]).expect("cast failed");
            if let Some(flag) = flags_array.iter().next().unwrap_or(None) {
                format!("(?{})", flag)
            } else {
                String::new()
            }
        } else {
            String::new()
        };        
    
        let array = input.into_iter().zip(pattern.into_iter().zip(replacement.into_iter())).map(|(text, (pat, rep))| {
            if let (Some(text), Some(pat), Some(rep)) = (text, pat, rep) {
                let prefix = &text[..start.unwrap_or(0) as usize];
                let rest_text = &text[start.unwrap_or(0) as usize..];
                let re = Regex::new(&format!("{}{}", flag_string, pat)).expect("Invalid regex pattern");
                
                if n == Some(0) {
                    // 替换所有匹配
                    Some(format!("{}{}", prefix, re.replace_all(rest_text, rep)))
                } else {
                    let mut replaced_text = String::from(prefix);
                    let mut count = 0;
                    let mut last_end = 0;
                    for cap in re.captures_iter(rest_text) {
                        count += 1;
                        replaced_text.push_str(&rest_text[last_end..cap.get(0).unwrap().start()]);
                        if count == n.unwrap() {
                            replaced_text.push_str(rep);
                        } else {
                            replaced_text.push_str(&cap.get(0).unwrap().as_str());
                        }
                        last_end = cap.get(0).unwrap().end();
                    }
                    replaced_text.push_str(&rest_text[last_end..]);
                    Some(replaced_text)
                }
            } else {
                None
            }
        }).collect::<StringArray>();        
        
        Ok(Arc::new(array) as ArrayRef)
    }    
}

// Function package declaration
pub struct FunctionPackage;

impl ScalarFunctionPackage for FunctionPackage {
    fn functions(&self) -> Vec<Box<dyn ScalarFunctionDef>> {
        vec![Box::new(RegexpCountFunction), Box::new(RegexpLikeFunction),Box::new(RegexpReplaceFunction)]
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
    async fn test_regexp_count() -> Result<()> {
        test_expression!("regexp_count('123456789012', '\\d\\d\\d', 2)", "3");
        test_expression!("regexp_count('abcabcabc', 'abc')", "3");
        test_expression!("regexp_count('AaBaHHACcAa', '[Aa]', 2)", "4");
        Ok(())
    }
    #[tokio::test]
    async fn test_regexp_like() -> Result<()> {
        test_expression!("regexp_like('abcabcabc', 'abc')", "true");
        test_expression!("regexp_like('Hello World', 'world$', 'i')", "true");
        Ok(())
    }        

    #[tokio::test]
    async fn test_regexp_replace() -> Result<()> {
        // 测试第二个匹配的'.'替换为'X'
        test_expression!("regexp_replace('Thomas', '.', 'X', 3, 2)", "ThoXas");

        // 测试从第2个字符开始, 替换所有的'a'为'X'
        test_expression!("regexp_replace('banana', 'a', 'X', 1, 0)", "bXnXnX");

        // 测试从第2个字符开始, 替换第1个匹配的'a'为'X'
        test_expression!("regexp_replace('banana', 'a', 'X', 2, 1)", "bXnana");

        // 测试不考虑大小写，从第2个字符开始, 替换第1个匹配的'a'或'A'为'X'
        test_expression!("regexp_replace('bAnana', 'a', 'X', 2, 1, 'i')", "bXnana");

        Ok(())
    }

}