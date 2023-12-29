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
    datatypes::{DataType, Date32Type, IntervalDayTimeType, IntervalUnit, TimeUnit},
};
use chrono::{Datelike, Duration, Local, NaiveDate, Offset, TimeZone, Timelike, Utc, NaiveDateTime};
use datafusion::error::Result;
use datafusion_common::DataFusionError;
use datafusion_expr::{
    ReturnTypeFunction, ScalarFunctionDef, ScalarFunctionPackage, Signature,
    TypeSignature, Volatility,
};

use arrow::array::*;
use arrow::error::ArrowError;

#[derive(Debug)]
pub struct HumanReadableSecondsFunction;

impl ScalarFunctionDef for HumanReadableSecondsFunction {
    fn name(&self) -> &str {
        "human_readable_seconds"
    }

    fn signature(&self) -> Signature {
        Signature::exact(vec![DataType::Float64], Volatility::Immutable)
    }

    fn return_type(&self) -> ReturnTypeFunction {
        let return_type = Arc::new(DataType::Utf8);
        Arc::new(move |_| Ok(return_type.clone()))
    }

    fn execute(&self, args: &[ArrayRef]) -> Result<ArrayRef> {
        assert_eq!(args.len(), 1);
        let input_array = args[0]
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("cast to Float64Array failed");

        let array = input_array
            .into_iter()
            .map(|sec| {
                let seconds = sec.map(|value| value).unwrap();
                let weeks = (seconds / 604800.0) as i64;
                let days = ((seconds % 604800.0) / 86400.0) as i64;
                let hours = ((seconds % 86400.0) / 3600.0) as i64;
                let minutes = ((seconds % 3600.0) / 60.0) as i64;
                let seconds_remainder = (seconds % 60.0) as i64;

                let mut formatted = String::new();

                if weeks > 0 {
                    formatted += &format!(
                        "{} week{}{}",
                        weeks,
                        if weeks > 1 { "s" } else { "" },
                        if days + hours + minutes + seconds_remainder > 0 {
                            ", "
                        } else {
                            ""
                        }
                    ); //for splitting ,
                }
                if days > 0 {
                    formatted += &format!(
                        "{} day{}{}",
                        days,
                        if days > 1 { "s" } else { "" },
                        if hours + minutes + seconds_remainder > 0 {
                            ", "
                        } else {
                            ""
                        }
                    ); //for splitting ,
                }
                if hours > 0 {
                    formatted += &format!(
                        "{} hour{}{}",
                        hours,
                        if hours > 1 { "s" } else { "" },
                        if minutes + seconds_remainder > 0 {
                            ", "
                        } else {
                            ""
                        }
                    ); //for splitting ,
                }
                if minutes > 0 {
                    formatted += &format!(
                        "{} minute{}{}",
                        minutes,
                        if minutes > 1 { "s" } else { "" },
                        if seconds_remainder > 0 { ", " } else { "" }
                    );
                }
                if seconds_remainder > 0 {
                    formatted += &format!(
                        "{} second{}",
                        seconds_remainder,
                        if seconds_remainder > 1 { "s" } else { "" }
                    );
                }
                if weeks + days + hours + minutes + seconds_remainder == 0 {
                    formatted = "0 second".to_string();
                }
                Some(formatted)
            })
            .collect::<StringArray>();
        Ok(Arc::new(array) as ArrayRef)
    }
}

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

#[derive(Debug)]
pub struct ToMilliSecondsFunction;

impl ScalarFunctionDef for ToMilliSecondsFunction {
    fn name(&self) -> &str {
        "to_milliseconds"
    }

    fn signature(&self) -> Signature {
        Signature::exact(
            vec![DataType::Interval(IntervalUnit::MonthDayNano)],
            Volatility::Immutable,
        )
    }

    fn return_type(&self) -> ReturnTypeFunction {
        let return_type = Arc::new(DataType::Int64);
        Arc::new(move |_| Ok(return_type.clone()))
    }

    fn execute(&self, args: &[ArrayRef]) -> Result<ArrayRef> {
        let input_array = args[0]
            .as_any()
            .downcast_ref::<IntervalMonthDayNanoArray>()
            .expect("cast to MonthDayNanoArray");

        let array = input_array
            .iter()
            .map(|arg| {
                let value = arg.unwrap() as u128;
                let months_part: i32 =
                    ((value & 0xFFFFFFFF000000000000000000000000) >> 96) as i32;
                    assert!(months_part == 0, "Error: You try to use Trino to_milliseconds(days-seconds). months must be zero");
                let days_part: i32 = ((value & 0xFFFFFFFF0000000000000000) >> 64) as i32;
                let nanoseconds_part: i64 = (value & 0xFFFFFFFFFFFFFFFF) as i64;
                let milliseconds:i64 = (days_part * 24*60*60*1000).into();
                let milliseconds= milliseconds + nanoseconds_part / 1_000_000;
                Some(milliseconds)
            })
            .collect::<Vec<_>>();
        let array = Int64Array::from(array);
        Ok(Arc::new(array) as ArrayRef)
    }
}

#[derive(Debug)]
pub struct ToIso8601Function;

impl ScalarFunctionDef for ToIso8601Function {
    fn name(&self) -> &str {
        "to_iso8601"
    }

    fn signature(&self) -> Signature {
        // This function accepts a Date, Timestamp, or Timestamp with TimeZone
        Signature::one_of(
            vec![
                TypeSignature::Exact(vec![DataType::Date32]),
                TypeSignature::Exact(vec![DataType::Timestamp(
                    TimeUnit::Nanosecond,
                    None,
                )]),
                TypeSignature::Exact(vec![DataType::Utf8]),
            ],
            Volatility::Immutable,
        )
    }

    fn return_type(&self) -> ReturnTypeFunction {
        let return_type = Arc::new(DataType::Utf8);
        Arc::new(move |_| Ok(return_type.clone()))
    }

    fn execute(&self, args: &[ArrayRef]) -> Result<ArrayRef> {
        if args.is_empty() {
            return Err(ArrowError::InvalidArgumentError(
                "args is empty".to_string(),
            ))
            .map_err(|err| DataFusionError::Execution(format!("Cast error: {}", err)));
        }

        let input = &args[0];
        let result = match input.data_type() {
            DataType::Date32 => {
                // Assuming the first array in args is a DateArray
                let date_array = args[0]
                    .as_any()
                    .downcast_ref::<Date32Array>()
                    .expect("Expected a date array");

                // Map each date32 to an ISO 8601 string
                let iso_strings: Vec<String> = date_array
                    .iter()
                    .map(|date| {
                        date.map(|date| {
                            let naive_date = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap()
                                + Duration::days(date as i64);
                            naive_date.format("%Y-%m-%d").to_string()
                        })
                        .unwrap_or_else(|| String::from("null")) // Handle null values
                    })
                    .collect();

                // Create a new StringArray from the ISO 8601 strings
                Ok(Arc::new(StringArray::from(iso_strings)) as Arc<dyn Array>)
            }
            DataType::Timestamp(TimeUnit::Nanosecond, None) => {
                let timestamp_array = input
                    .as_any()
                    .downcast_ref::<TimestampNanosecondArray>()
                    .expect("Expected a NanosecondTimeStamp array");
                let milliseconds_array = timestamp_array
                    .iter()
                    .map(|timestamp| timestamp.map(|timestamp| timestamp / 1_000_000))
                    .collect::<TimestampMillisecondArray>();

                let iso_strings: Vec<String> = milliseconds_array
                    .iter()
                    .map(|timestamp| {
                        timestamp
                            .map(|timestamp| {
                                let datetime =
                                    Utc.timestamp_millis_opt(timestamp).unwrap();
                                format!("{}", datetime.format("%Y-%m-%dT%H:%M:%S%.3f"))
                            })
                            .unwrap_or_else(|| String::from("null")) // Handle null values
                    })
                    .collect();

                Ok(Arc::new(StringArray::from(iso_strings)) as Arc<dyn Array>)
            }
            // timestamp with timezone todo
            _ => Err(ArrowError::InvalidArgumentError(
                "Invalid input type".to_string(),
            ))
            .map_err(|err| DataFusionError::Execution(format!("Cast error: {}", err))),
        };
        result
    }
}
#[derive(Debug)]
pub struct FromIso8601DateFunction;
///presto `from_iso8601_date` function ([https://trino.io/docs/current/functions/datetime.html])
impl ScalarFunctionDef for FromIso8601DateFunction {
    fn name(&self) -> &str {
        "from_iso8601_date"
    }

    fn signature(&self) -> Signature {
        Signature::exact(vec![DataType::Utf8], Volatility::Immutable)
    }

    fn return_type(&self) -> ReturnTypeFunction {
        let return_type = Arc::new(DataType::Date32);
        Arc::new(move |_| Ok(return_type.clone()))
    }

    fn execute(&self, args: &[ArrayRef]) -> Result<ArrayRef> {
        let input = &args[0]
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("Failed to downcast StringArray");
        let mut result = Vec::new();

        for i in 0..input.len() {
            if input.is_null(i) {
                result.push(None);
            } else {
                let value = input.value(i);
                let parsed_date = NaiveDate::parse_from_str(value, "%Y-%m-%d")
                    .or_else(|_| NaiveDate::parse_from_str(value, "%G-W%V-%u"))
                    .or_else(|_| {
                        NaiveDate::parse_from_str(&format!("{}-1", value), "%G-W%V-%u")
                    })
                    .or_else(|_| NaiveDate::parse_from_str(value, "%Y-%j"));

                match parsed_date {
                    Ok(date) => result.push(Some(date.num_days_from_ce() - 719163)), // Adjust for Unix time
                    Err(_) => result.push(None),
                }
            }
        }

        Ok(Arc::new(Date32Array::from(result)) as ArrayRef)
    }
}

#[derive(Debug)]
pub struct UnixTimeFunction;

impl ScalarFunctionDef for UnixTimeFunction {
    fn name(&self) -> &str {
        "to_unixtime"
    }

    fn signature(&self) -> Signature {
        Signature::exact(
            vec![DataType::Timestamp(TimeUnit::Nanosecond, None)],
            Volatility::Immutable,
        )
    }

    fn return_type(&self) -> ReturnTypeFunction {
        Arc::new(move |_| Ok(Arc::new(DataType::Float64)))
    }

    fn execute(&self, args: &[ArrayRef]) -> Result<ArrayRef> {
        assert_eq!(args.len(), 1);
        let timestamp_array = args[0]
            .as_any()
            .downcast_ref::<TimestampNanosecondArray>()
            .expect("cast to TimestampNanosecondArray failed");

        let mut b = Float64Builder::with_capacity(timestamp_array.len());

        for i in 0..timestamp_array.len() {
            if timestamp_array.is_null(i) {
                b.append_null();
                continue;
            }

            let timestamp_value = timestamp_array.value(i);
            // Convert nanoseconds to seconds
            let unixtime = (timestamp_value as f64) / 1_000_000_000.0;
            b.append_value(unixtime);
        }

        Ok(Arc::new(b.finish()))
    }
}

#[derive(Debug)]
pub struct FromUnixtimeFunction;

impl ScalarFunctionDef for FromUnixtimeFunction {
    fn name(&self) -> &str {
        "from_unixtime"
    }

    fn signature(&self) -> Signature {
        Signature::one_of(
            vec![
                TypeSignature::Exact(vec![DataType::Int64]),
                //TypeSignature::Exact(vec![DataType::Int64, DataType::Utf8]),
            ],
            Volatility::Immutable,
        )
    }

    fn return_type(&self) -> ReturnTypeFunction {
        let return_type = Arc::new(DataType::Timestamp(TimeUnit::Second, None));
        Arc::new(move |_| Ok(return_type.clone()))
    }

    fn execute(&self, args: &[ArrayRef]) -> Result<ArrayRef> {
        assert_eq!(args.len(), 1);
        let unixtime_array = args[0]
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("cast to Int64Array failed");

        let mut builder = TimestampSecondArray::builder(unixtime_array.len());

        for i in 0..unixtime_array.len() {
            if unixtime_array.is_null(i) {
                builder.append_null();
                continue;
            }

            let unixtime_value = unixtime_array.value(i);
            builder.append_value(unixtime_value);
        }

        Ok(Arc::new(builder.finish()))
    }
}

#[derive(Debug)]
pub struct FromUnixtimeNanosFunction;

impl ScalarFunctionDef for FromUnixtimeNanosFunction {
    fn name(&self) -> &str {
        "from_unixtime_nanos"
    }

    fn signature(&self) -> Signature {
        Signature::exact(vec![DataType::Int64], Volatility::Immutable)
    }

    fn return_type(&self) -> ReturnTypeFunction {
        let return_type = Arc::new(DataType::Timestamp(TimeUnit::Nanosecond, None));
        Arc::new(move |_| Ok(return_type.clone()))
    }

    fn execute(&self, args: &[ArrayRef]) -> Result<ArrayRef> {
        assert_eq!(args.len(), 1);
        let unixtime_array = args[0]
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("cast to Int64Array failed");

        let mut builder = TimestampNanosecondArray::builder(unixtime_array.len());

        for i in 0..unixtime_array.len() {
            if unixtime_array.is_null(i) {
                builder.append_null();
                continue;
            }

            let unixtime_value = unixtime_array.value(i);
            builder.append_value(unixtime_value);
        }

        Ok(Arc::new(builder.finish()))
    }
}

#[derive(Debug)]
pub struct DayFunction;

impl ScalarFunctionDef for DayFunction {
    fn name(&self) -> &str {
        "day"
    }

    fn signature(&self) -> Signature {
        // Function accepts Date, Interval Day to Second, or Timestamp
        Signature::one_of(
            vec![
                TypeSignature::Exact(vec![DataType::Date32]),
                TypeSignature::Exact(vec![DataType::Interval(IntervalUnit::DayTime)]),
                TypeSignature::Exact(vec![DataType::Timestamp(
                    TimeUnit::Nanosecond,
                    None,
                )]),
            ],
            Volatility::Immutable,
        )
    }

    fn return_type(&self) -> ReturnTypeFunction {
        let return_type = Arc::new(DataType::Int64);
        Arc::new(move |_| Ok(return_type.clone()))
    }

    fn execute(&self, args: &[ArrayRef]) -> Result<ArrayRef> {
        assert_eq!(args.len(), 1);

        let input = &args[0];
        let result = match input.data_type() {
            DataType::Date32 => {
                let date_array = input
                    .as_any()
                    .downcast_ref::<Date32Array>()
                    .expect("Expected a date array");

                let days: Vec<i64> = date_array
                    .iter()
                    .map(|date_opt| {
                        date_opt
                            .map(|date| {
                                let naive_date = Date32Type::to_naive_date(date);

                                naive_date.day() as i64
                            })
                            .unwrap_or(0)
                    })
                    .collect();

                Ok(Arc::new(Int64Array::from(days)) as ArrayRef)
            }
            DataType::Interval(IntervalUnit::DayTime) => {
                let interval_array = input
                    .as_any()
                    .downcast_ref::<IntervalDayTimeArray>()
                    .expect("Expected an interval day to second array");

                let days: Vec<i64> = interval_array
                    .iter()
                    .map(|interval_opt| {
                        interval_opt
                            .map(|interval| {
                                let (days, _ms) = IntervalDayTimeType::to_parts(interval);
                                days as i64
                            })
                            .unwrap_or(0)
                    })
                    .collect();

                Ok(Arc::new(Int64Array::from(days)) as ArrayRef)
            }
            DataType::Timestamp(TimeUnit::Nanosecond, _) => {
                let timestamp_array = input
                    .as_any()
                    .downcast_ref::<TimestampNanosecondArray>()
                    .expect("Expected a nanosecond timestamp array");

                let days: Vec<i64> = timestamp_array
                    .iter()
                    .map(|timestamp_opt| {
                        timestamp_opt
                            .map(|timestamp| {
                                let datetime = NaiveDateTime::from_timestamp_opt(
                                    timestamp / 1_000_000_000,
                                    (timestamp % 1_000_000_000) as u32,
                                );

                                datetime.map(|dt| dt.date().day() as i64).unwrap_or(0)
                            })
                            .unwrap_or(0)
                    })
                    .collect();

                Ok(Arc::new(Int64Array::from(days)) as ArrayRef)
            }
            _ => Err(ArrowError::InvalidArgumentError(
                "Invalid input type".to_string(),
            )),
        };
        Ok(result?)
    }
}

// Function package declaration
pub struct FunctionPackage;

impl ScalarFunctionPackage for FunctionPackage {
    fn functions(&self) -> Vec<Box<dyn ScalarFunctionDef>> {
        vec![
            Box::new(HumanReadableSecondsFunction),
            Box::new(CurrentTimeFunction),
            Box::new(ToMilliSecondsFunction),
            Box::new(CurrentTimestampFunction),
            Box::new(CurrentTimestampPFunction),
            Box::new(ToIso8601Function),
            Box::new(FromIso8601DateFunction),
            Box::new(UnixTimeFunction),
            Box::new(FromUnixtimeFunction),
            Box::new(FromUnixtimeNanosFunction),
            Box::new(CurrentTimezoneFunction),
            Box::new(LocaltimeFunction),
            Box::new(LocaltimestampFunction),
            Box::new(LocaltimestampPFunction),
            Box::new(DayFunction),
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
    #[tokio::test]
    async fn test_human_readable_seconds() -> Result<()> {
        test_expression!("human_readable_seconds(604800.0)", "1 week");
        test_expression!("human_readable_seconds(86400.0)", "1 day");
        test_expression!("human_readable_seconds(3600.0)", "1 hour");
        test_expression!("human_readable_seconds(60.0)", "1 minute");
        test_expression!("human_readable_seconds(1.0)", "1 second");
        test_expression!("human_readable_seconds(0.0)", "0 second");
        test_expression!("human_readable_seconds(96)", "1 minute, 36 seconds");
        test_expression!(
            "human_readable_seconds(3762)",
            "1 hour, 2 minutes, 42 seconds"
        );
        test_expression!(
            "human_readable_seconds(56363463)",
            "93 weeks, 1 day, 8 hours, 31 minutes, 3 seconds"
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_to_iso8601() -> Result<()> {
        //Test cases for different input types
        // Date
        test_expression!("to_iso8601(Date '2023-03-15')", "2023-03-15");
        // Timestamp
        test_expression!(
            "to_iso8601( timestamp '2001-04-13T02:00:00')",
            "2001-04-13T02:00:00.000"
        );
        //TIMESTAMP '2020-06-10 15:55:23.383345'
        test_expression!(
            "to_iso8601( timestamp '2020-06-10 15:55:23.383345')",
            "2020-06-10T15:55:23.383"
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_from_iso8601_date() -> Result<()> {
        test_expression!("from_iso8601_date('2020-05-11')", "2020-05-11");
        test_expression!("from_iso8601_date('2020-W10')", "2020-03-02");
        test_expression!("from_iso8601_date('2020-W10-1')", "2020-03-02");
        test_expression!("from_iso8601_date('2020-123')", "2020-05-02");
        Ok(())
    }

    #[tokio::test]
    async fn test_to_unixtime() -> Result<()> {
        test_expression!(
            "to_unixtime(Date '2023-03-15')",
            "1678838400.0" // UNIX timestamp for 2023-03-15 00:00:00 UTC
        );

        // Test case for a specific timestamp without sub-second precision
        test_expression!(
            "to_unixtime(timestamp '2001-04-13T02:00:00')",
            "987127200.0" // UNIX timestamp for 2001-04-13 02:00:00 UTC
        );

        // Test case for a specific timestamp with sub-second precision
        test_expression!(
            "to_unixtime(timestamp '2020-06-10 15:55:23.383345')",
            "1591804523.383345" // UNIX timestamp for 2020-06-10 15:55:23.383345 UTC
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_from_unixtime() -> Result<()> {
        test_expression!(
            "from_unixtime(1591804523)",
            "2020-06-10T15:55:23" //"2020-06-10 15:55:23.000"
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_from_unixtime_nanos() -> Result<()> {
        test_expression!(
            "from_unixtime_nanos(1591804523000000000)",
            "2020-06-10T15:55:23" //"2020-06-10 15:55:23.000"
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_day() -> Result<()> {
        test_expression!("day(Date '2023-03-15')", "15");
        //test_expression!("day(Interval '10' DAY)", "10");
        //test_expression!("day(Interval '-20' DAY)", "-20");
        test_expression!("day(timestamp '2001-04-13T02:00:00')", "13");
        test_expression!("day(timestamp '2020-06-10 15:55:23.383345')", "10");
        Ok(())
    }
}
