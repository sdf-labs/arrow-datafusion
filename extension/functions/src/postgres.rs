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
 
use arrow::array::{ArrayRef, StringArray};
use arrow::datatypes::DataType;
use datafusion::error::Result;
use datafusion::logical_expr::Volatility;
use datafusion_common::cast::{as_string_array};
use datafusion_expr::{
    ReturnTypeFunction, ScalarFunctionDef, ScalarFunctionPackage, Signature,
};
use std::sync::Arc;

#[derive(Debug)]
pub struct FormatFunction;

impl ScalarFunctionDef for FormatFunction {
    fn name(&self) -> &str {
        "format"
    }

    fn signature(&self) -> Signature {
        Signature::variadic(vec![DataType::Utf8], Volatility::Immutable)
    }

    fn return_type(&self) -> ReturnTypeFunction {
        let return_type = Arc::new(DataType::Utf8);
        Arc::new(move |_| Ok(return_type.clone()))
    }

    fn execute(&self, args: &[ArrayRef]) -> Result<ArrayRef> {
        assert!(args.len() >= 1);
        
        let format_array = as_string_array(&args[0]).expect("cast failed");
        let args_array: Vec<_> = args[1..].iter().map(|arg| as_string_array(arg).expect("cast failed")).collect();

        let result_array: Vec<Option<String>> = format_array.into_iter().enumerate().map(|(idx, format)| {
            if let Some(format) = format {
                let mut result = format.clone();
                for (i, arg_array) in args_array.iter().enumerate() {
                    let replace_str = format!("%{}$s", i + 1);
                    let replace_with = arg_array.value(idx);
                    // Simply replacing the contents of result
                    result = &result.replace(&replace_str, &replace_with);
                }                
                Some(result)
            } else {
                None
            }
        }).collect();

        Ok(Arc::new(StringArray::from(result_array)) as ArrayRef)
    }
}


// Function package declaration
pub struct FunctionPackage;

impl ScalarFunctionPackage for FunctionPackage {
    fn functions(&self) -> Vec<Box<dyn ScalarFunctionDef>> {
        vec![Box::new(FormatFunction)]
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
    async fn test_format() -> Result<()> {
        test_expression!("format('Hello %s, %1$s', 'World')", "'Hello World, World'");
        // Add more test cases as needed.
        Ok(())
    }    

}