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

use std::sync::Arc;

use arrow::{datatypes::{DataType, Date32Type}, array::{ArrayRef, PrimitiveBuilder, PrimitiveArray, Int32Array, Date32Array}};
use chrono::{Utc, Datelike};
use datafusion_expr::{ScalarFunctionPackage, ScalarFunctionDef, Signature, Volatility, ReturnTypeFunction};

use datafusion::error::Result;

#[derive(Debug)]
pub struct CurrentDateFunction;

impl ScalarFunctionDef for CurrentDateFunction {
    fn name(&self) -> &str {
        "current_date"
    }

    fn signature(&self) -> Signature {
        Signature::exact(vec![], Volatility::Immutable)
    }

    fn return_type(&self) -> ReturnTypeFunction {
        let return_type = Arc::new(DataType::Date32);
        Arc::new(move |_| Ok(return_type.clone()))
    }

    fn execute(&self, _args: &[ArrayRef]) -> Result<ArrayRef> {
        // let current_date = chrono::Utc::now().naive_utc();
        // let date_value = current_date.date();
        // let millis_since_epoch = date_value.and_hms_opt(0, 0, 0).unwrap().timestamp_millis();
        // let array = Date32Array::from(vec![Some(millis_since_epoch as i32)]);
        let current_date = chrono::Utc::now().naive_utc();
        let date_value = current_date.date();
        let days_since_epoch = date_value.signed_duration_since(chrono::NaiveDate::from_ymd_opt(1970, 1, 1).unwrap()).num_days();
        let array = Date32Array::from(vec![Some(days_since_epoch as i32)]);
        Ok(Arc::new(array) as ArrayRef)
    }
}

// Function package declaration
pub struct FunctionPackage;

impl ScalarFunctionPackage for FunctionPackage {
    fn functions(&self) -> Vec<Box<dyn ScalarFunctionDef>> {
        vec![Box::new(CurrentDateFunction)]
    }
}

#[cfg(test)]
mod test {
    use datafusion::error::Result;
    use datafusion::prelude::SessionContext;
    use tokio;

    use crate::utils::{execute, test_expression};

    use super::FunctionPackage;

    use super::*;
    use chrono::offset::TimeZone;
    use chrono::offset::Utc;
    use chrono::Datelike;
    use arrow::array::Date32Array;

    #[tokio::test]
    async fn test_current_date() -> Result<()> {
        test_expression!(
            "current_date()",
            "2023-11-26"
        );
        Ok(())
    }
}