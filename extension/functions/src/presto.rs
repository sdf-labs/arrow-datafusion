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

use std::{
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use arrow::{
    array::{
        ArrayRef, Int64Array, StringArray, Time32MillisecondArray,
        TimestampMillisecondArray, TimestampNanosecondArray,
    },
    datatypes::{DataType, TimeUnit},
};
use chrono::{Local, Offset, Timelike, Utc};
use datafusion::error::Result;
use datafusion_common::DataFusionError;
use datafusion_expr::{
    ReturnTypeFunction, ScalarFunctionDef, ScalarFunctionPackage, Signature, Volatility,
};

#[derive(Debug)]
pub struct CurrentTimeFunction;

impl ScalarFunctionDef for CurrentTimeFunction {
    fn name(&self) -> &str {
        "current_time"
    }

    fn signature(&self) -> Signature {
        Signature::exact(vec![], Volatility::Immutable)
    }

    fn return_type(&self) -> ReturnTypeFunction {
        let return_type = Arc::new(DataType::Time32(TimeUnit::Millisecond));
        Arc::new(move |_| Ok(return_type.clone()))
    }

    fn execute(&self, _args: &[ArrayRef]) -> Result<ArrayRef> {
        let current_time = chrono::Utc::now();
        let milliseconds_since_midnight = current_time.num_seconds_from_midnight() * 1000;
        let array =
            Time32MillisecondArray::from(vec![Some(milliseconds_since_midnight as i32)]);
        Ok(Arc::new(array) as ArrayRef)
    }
}
#[derive(Debug)]
pub struct CurrentTimestampFunction;

impl ScalarFunctionDef for CurrentTimestampFunction {
    fn name(&self) -> &str {
        "current_timestamp"
    }

    fn signature(&self) -> Signature {
        Signature::exact(vec![], Volatility::Immutable)
    }

    fn return_type(&self) -> ReturnTypeFunction {
        Arc::new(move |_| Ok(Arc::new(DataType::Timestamp(TimeUnit::Millisecond, None))))
    }

    fn execute(&self, _args: &[ArrayRef]) -> Result<ArrayRef> {
        let now = Utc::now();
        let milliseconds_since_epoch = now.timestamp_millis();
        let array = TimestampMillisecondArray::from(vec![Some(milliseconds_since_epoch)]);
        Ok(Arc::new(array) as ArrayRef)
    }
}

#[derive(Debug)]
pub struct CurrentTimestampPFunction;

impl ScalarFunctionDef for CurrentTimestampPFunction {
    fn name(&self) -> &str {
        "current_timestamp_p"
    }

    fn signature(&self) -> Signature {
        Signature::exact(vec![DataType::Int64], Volatility::Immutable)
    }

    fn return_type(&self) -> ReturnTypeFunction {
        let return_type = Arc::new(DataType::Timestamp(TimeUnit::Nanosecond, None));
        Arc::new(move |_| Ok(return_type.clone()))
    }

    fn execute(&self, args: &[ArrayRef]) -> Result<ArrayRef> {
        let precision = match args.get(0) {
            Some(array) => {
                let array = array.as_any().downcast_ref::<Int64Array>().unwrap();
                array.value(0) as usize
            }
            None => 6,
        };

        if precision > 9 {
            return Err(DataFusionError::Execution(
                "Precision greater than nanoseconds is not supported".to_string(),
            ));
        }

        let now = chrono::Utc::now();
        let nanos = now.timestamp_subsec_nanos();
        let timestamp = now.timestamp() as i64 * 1_000_000_000 + nanos as i64;

        let divisor = 10_i64.pow(9 - precision as u32);
        let adjusted_timestamp = (timestamp / divisor) * divisor;

        let array = TimestampNanosecondArray::from(vec![Some(adjusted_timestamp)]);
        Ok(Arc::new(array) as ArrayRef)
    }
}

#[derive(Debug)]
pub struct CurrentTimezoneFunction;

impl ScalarFunctionDef for CurrentTimezoneFunction {
    fn name(&self) -> &str {
        "current_timezone"
    }

    fn signature(&self) -> Signature {
        Signature::exact(vec![], Volatility::Immutable)
    }

    fn return_type(&self) -> ReturnTypeFunction {
        Arc::new(move |_| Ok(Arc::new(DataType::Utf8)))
    }

    fn execute(&self, _args: &[ArrayRef]) -> Result<ArrayRef> {
        let now_local = Local::now();
        let timezone = format!("{}", now_local.offset().fix());
        let array = StringArray::from(vec![Some(timezone)]);
        Ok(Arc::new(array) as ArrayRef)
    }
}

#[derive(Debug)]
pub struct LocaltimeFunction;

impl ScalarFunctionDef for LocaltimeFunction {
    fn name(&self) -> &str {
        "localtime"
    }
    fn signature(&self) -> Signature {
        Signature::exact(vec![], Volatility::Immutable)
    }

    fn return_type(&self) -> ReturnTypeFunction {
        let return_type = Arc::new(DataType::Time32(TimeUnit::Millisecond));
        Arc::new(move |_| Ok(return_type.clone()))
    }

    fn execute(&self, _args: &[ArrayRef]) -> Result<ArrayRef> {
        let local_time = chrono::Local::now().time();
        let milliseconds_since_midnight = local_time.num_seconds_from_midnight() * 1000;
        let array =
            Time32MillisecondArray::from(vec![Some(milliseconds_since_midnight as i32)]);
        Ok(Arc::new(array) as ArrayRef)
    }
}
#[derive(Debug)]
pub struct LocaltimestampFunction;

impl ScalarFunctionDef for LocaltimestampFunction {
    fn name(&self) -> &str {
        "localtimestamp"
    }

    fn signature(&self) -> Signature {
        Signature::exact(vec![], Volatility::Immutable)
    }

    fn return_type(&self) -> ReturnTypeFunction {
        Arc::new(move |_| Ok(Arc::new(DataType::Timestamp(TimeUnit::Millisecond, None))))
    }

    fn execute(&self, _args: &[ArrayRef]) -> Result<ArrayRef> {
        let n = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|err| DataFusionError::Execution(err.to_string()))?;
        let array = TimestampMillisecondArray::from(vec![Some(n.as_millis() as i64)]);
        Ok(Arc::new(array) as ArrayRef)
    }
}

#[derive(Debug)]
pub struct LocaltimestampPFunction;

impl ScalarFunctionDef for LocaltimestampPFunction {
    fn name(&self) -> &str {
        "localtimestamp_p"
    }

    fn signature(&self) -> Signature {
        Signature::exact(vec![DataType::Int64], Volatility::Immutable)
    }

    fn return_type(&self) -> ReturnTypeFunction {
        Arc::new(move |_| Ok(Arc::new(DataType::Timestamp(TimeUnit::Nanosecond, None))))
    }

    fn execute(&self, args: &[ArrayRef]) -> Result<ArrayRef> {
        let precision = match args.get(0) {
            Some(array) => {
                let array = array.as_any().downcast_ref::<Int64Array>().unwrap();
                array.value(0) as usize
            }
            None => 6,
        };

        if precision > 9 {
            return Err(DataFusionError::Execution(
                "Precision greater than nanoseconds is not supported".to_string(),
            ));
        }

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|err| DataFusionError::Execution(err.to_string()))?;
        let nanos = now.as_nanos() as i64;

        let factor = 10_i64.pow(9 - precision as u32);
        let adjusted_timestamp = (nanos / factor) * factor;

        let array = TimestampNanosecondArray::from(vec![Some(adjusted_timestamp)]);
        Ok(Arc::new(array) as ArrayRef)
    }
}

// Function package declaration
pub struct FunctionPackage;

impl ScalarFunctionPackage for FunctionPackage {
    fn functions(&self) -> Vec<Box<dyn ScalarFunctionDef>> {
        vec![
            Box::new(CurrentTimeFunction),
            Box::new(CurrentTimestampFunction),
            Box::new(CurrentTimestampPFunction),
            Box::new(CurrentTimezoneFunction),
            Box::new(LocaltimeFunction),
            Box::new(LocaltimestampFunction),
            Box::new(LocaltimestampPFunction),
        ]
    }
}

#[cfg(test)]
mod test {
    use std::{
        sync::Arc,
        time::{SystemTime, UNIX_EPOCH},
    };

    use arrow::array::{
        Array, ArrayRef, Int64Array, TimestampMillisecondArray, TimestampNanosecondArray,
    };
    use chrono::{DateTime, Local, Offset, Utc};
    use datafusion::error::Result;
    use datafusion::prelude::SessionContext;
    use datafusion_expr::ScalarFunctionDef;
    use tokio;

    use crate::{
        presto::{
            CurrentTimestampFunction, CurrentTimestampPFunction, LocaltimestampFunction,
            LocaltimestampPFunction,
        },
        utils::{execute, test_expression},
    };

    use super::FunctionPackage;

    fn roughly_equal_to_now(millisecond: i64) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;
        (millisecond - now).abs() <= 1
    }

    fn test_timestamp<E: ScalarFunctionDef>(timestamp_func: E) -> Result<()> {
        let result = timestamp_func.execute(&[]).unwrap();
        let result_array = result
            .as_any()
            .downcast_ref::<TimestampMillisecondArray>()
            .unwrap()
            .value(0);
        assert!(roughly_equal_to_now(result_array));
        Ok(())
    }

    fn test_timestamp_p<E: ScalarFunctionDef>(timestamp_p_func: E) -> Result<()> {
        let precision_array = Int64Array::from(vec![9]);
        let args = vec![Arc::new(precision_array) as ArrayRef];
        let result = timestamp_p_func.execute(&args).unwrap();
        let result_array = result
            .as_any()
            .downcast_ref::<TimestampNanosecondArray>()
            .unwrap()
            .value(0);
        assert!(roughly_equal_to_now(result_array / 1_000_000));
        Ok(())
    }

    #[tokio::test]
    async fn test_current_time() -> Result<()> {
        let current = Utc::now();
        let formatted = current.format("%H:%M:%S").to_string();
        test_expression!("current_time()", formatted);
        Ok(())
    }

    #[tokio::test]
    async fn test_current_timestamp() -> Result<()> {
        test_timestamp(CurrentTimestampFunction {})?;
        Ok(())
    }

    #[tokio::test]
    async fn test_current_timestamp_p() -> Result<()> {
        test_timestamp_p(CurrentTimestampPFunction {})?;
        Ok(())
    }

    #[tokio::test]
    async fn test_current_timezone() -> Result<()> {
        let now_local: DateTime<Local> = Local::now();
        let timezone = format!("{}", now_local.offset().fix());
        test_expression!("current_timezone()", timezone);
        Ok(())
    }

    #[tokio::test]
    async fn test_localtime() -> Result<()> {
        let local = Local::now();
        let formatted = local.format("%H:%M:%S").to_string();
        test_expression!("localtime", formatted);
        Ok(())
    }

    #[tokio::test]
    async fn test_localtimestamp() -> Result<()> {
        test_timestamp(LocaltimestampFunction {})?;
        Ok(())
    }

    #[tokio::test]
    async fn test_localtimestamp_p() -> Result<()> {
        test_timestamp_p(LocaltimestampPFunction {})?;
        Ok(())
    }
}
