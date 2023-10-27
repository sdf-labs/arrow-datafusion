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

use arrow::array::{ArrayRef, Int64Array};
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
        let start = if args.len() > 2 {
            as_int64_array(&args[2]).expect("cast failed").iter().next().unwrap_or(None)
        } else {
            None
        };
    
        // 进行正则表达式匹配
        let array = input.into_iter().zip(pattern.into_iter()).map(|(text, pat)| {
            if let (Some(text), Some(pat)) = (text, pat) {
                // 如果提供了 start 参数，则从指定位置开始搜索
                let text = &text[start.unwrap_or(0) as usize..];
                // 创建正则表达式
                let re = Regex::new(&pat).expect("Invalid regex pattern");
                // 计算匹配次数
                // println!("Input text: {}", &text);
                // println!("Pattern: {}", &pat);
                // println!("Start index: {}", start.unwrap_or(0));

                // println!("Match count: {}", re.find_iter(text).count() as i64);
                Some(re.find_iter(text).count() as i64)
            } else {
                None
            }
        }).collect::<Int64Array>();
        
        // 返回结果
        Ok(Arc::new(array) as ArrayRef)
    }    
}

// Function package declaration
pub struct FunctionPackage;

impl ScalarFunctionPackage for FunctionPackage {
    fn functions(&self) -> Vec<Box<dyn ScalarFunctionDef>> {
        vec![Box::new(RegexpCountFunction)]
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

    
}