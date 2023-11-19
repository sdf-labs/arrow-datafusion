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
use arrow::datatypes::{self, IntervalUnit};
use arrow::datatypes::{DataType, TimeUnit};
use arrow::datatypes::{Field, IntervalDayTimeType};
use datafusion::error::Result;
// use datafusion::scalar::ScalarFunctionDef;
//use datafusion::physical_plan::functions::{Signature, Volatility};
use chrono::prelude::*;
use chrono::Months;
use chrono::{Datelike, Duration, NaiveDate, NaiveDateTime};
use datafusion::logical_expr::Volatility;
use datafusion::physical_plan::expressions::Column;
use datafusion_common::cast::{
    as_date32_array, as_float64_array, as_string_array, as_timestamp_nanosecond_array,
};
use datafusion_common::scalar::ScalarValue;
use datafusion_common::DataFusionError;
use datafusion_expr::ColumnarValue;
use datafusion_expr::{
    ReturnTypeFunction, ScalarFunctionDef, ScalarFunctionPackage, Signature,
};
use std::collections::HashSet;
use std::sync::Arc;

#[derive(Debug)]
pub struct JustifyHoursFunction;

impl ScalarFunctionDef for JustifyHoursFunction {
    fn name(&self) -> &str {
        "justify_hours"
    }

    fn signature(&self) -> Signature {
        Signature::exact(
            vec![
                DataType::Interval(IntervalUnit::DayTime),
            ],
            Volatility::Immutable,
        )
    }

    fn return_type(&self) -> ReturnTypeFunction {
        let return_type = Arc::new(DataType::Interval(IntervalUnit::DayTime));
        Arc::new(move |_| Ok(return_type.clone()))
    }

    fn execute(&self, args: &[ArrayRef]) -> Result<ArrayRef> {
        assert_eq!(args.len(), 1);

        let array = &args[0];
        let mut b = IntervalDayTimeBuilder::with_capacity(array.len());
        match array.data_type() {
            DataType::Interval(IntervalUnit::DayTime) => {
                let array = as_interval_day_time_array(&array)?;
                let iter: ArrayIter<&IntervalDayTimeArray> = ArrayIter::new(array);
                iter.into_iter().for_each(|value| {
                    if let Some(value) = value {
                        // Extract days, hours, and minutes from the interval
                        let days = value.days();
                        let hours = value.hours();
                        let minutes = value.minutes();

                        // Calculate the total number of minutes in the interval
                        let total_minutes = days * 24 * 60 + hours * 60 + minutes;

                        // Calculate the adjusted days and remaining minutes
                        let adjusted_days = total_minutes / (24 * 60);
                        let remaining_minutes = total_minutes % (24 * 60);

                        b.append_value(IntervalDayTimeType::make_value(adjusted_days, remaining_minutes));
                    } else {
                        b.append_null();
                    }
                });
            }

            _ => todo!(),
        }
        let result = b.finish();
        cast(
            &(Arc::new(result) as ArrayRef),
            &DataType::Interval(IntervalUnit::DayTime),
        )
        .map_err(|err| DataFusionError::Execution(format!("Cast error: {}", err)))
    }
}
// Function package declaration
pub struct FunctionPackage;

impl ScalarFunctionPackage for FunctionPackage {
    fn functions(&self) -> Vec<Box<dyn ScalarFunctionDef>> {
        vec![Box::new(JustifyHoursFunction)]
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
