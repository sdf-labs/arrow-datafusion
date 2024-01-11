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
    collections::HashMap,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use arrow::{
    array::{
        ArrayRef, Int64Array, Time32MillisecondArray, TimestampMillisecondArray,
        TimestampNanosecondArray,
    },
    datatypes::{DataType, Date32Type, IntervalDayTimeType, IntervalUnit, TimeUnit},
};

use chrono::{
    DateTime, Datelike, Duration, Local, Months, NaiveDate, NaiveDateTime, NaiveTime,
    Offset, TimeZone, Timelike, Utc, Weekday,
};
use datafusion::error::Result;
use datafusion_common::DataFusionError;
use datafusion_expr::{
    ReturnTypeFunction, ScalarFunctionDef, ScalarFunctionPackage, Signature,
    TypeSignature, Volatility,
};

use arrow::array::*;
use arrow::error::ArrowError;
use regex::Regex;

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

#[derive(Debug)]
pub struct DateTruncFunction;

impl ScalarFunctionDef for DateTruncFunction {
    fn name(&self) -> &str {
        "date_trunc"
    }

    fn signature(&self) -> Signature {
        Signature::one_of(
            vec![
                TypeSignature::Exact(vec![DataType::Utf8, DataType::Date32]),
                TypeSignature::Exact(vec![
                    DataType::Utf8,
                    DataType::Timestamp(TimeUnit::Nanosecond, None),
                ]),
                TypeSignature::Exact(vec![
                    DataType::Utf8,
                    DataType::Time64(TimeUnit::Nanosecond),
                ]),
            ],
            Volatility::Immutable,
        )
    }
    fn return_type(&self) -> ReturnTypeFunction {
        Arc::new(move |args| Ok(Arc::new(args[1].clone())))
    }

    fn execute(&self, args: &[ArrayRef]) -> Result<ArrayRef> {
        assert_eq!(args.len(), 2);

        // Extract arguments
        let unit = args[0].as_any().downcast_ref::<StringArray>().unwrap();
        let timestamps = &args[1];
        let result = match timestamps.data_type() {
            DataType::Date32 => {
                let date_array = args[1]
                    .as_any()
                    .downcast_ref::<Date32Array>()
                    .expect("Expected a Date32 array");
                let mut result = Vec::new();

                for i in 0..date_array.len() {
                    if date_array.is_null(i) {
                        result.push(None);
                        continue;
                    }

                    let date_value = date_array.value(i);
                    let naive_date =
                        NaiveDate::from_num_days_from_ce_opt(date_value + 719163)
                            .unwrap(); // converting to NaiveDate

                    let trunc = match unit.value(i).to_string().as_str() {
                        "day" => NaiveDate::from_ymd_opt(
                            naive_date.year(),
                            naive_date.month(),
                            naive_date.day(),
                        )
                        .unwrap(),
                        "month" => NaiveDate::from_ymd_opt(
                            naive_date.year(),
                            naive_date.month(),
                            1,
                        )
                        .unwrap(),
                        "year" => {
                            NaiveDate::from_ymd_opt(naive_date.year(), 1, 1).unwrap()
                        }
                        "week" => {
                            let weekday =
                                naive_date.weekday().num_days_from_monday() as i64;

                            naive_date - Duration::days(weekday)
                        }
                        "quarter" => {
                            let month = ((naive_date.month() - 1) / 3) * 3 + 1;
                            NaiveDate::from_ymd_opt(naive_date.year(), month, 1).unwrap()
                        }
                        _ => naive_date,
                    };
                    let days_since_epoch = trunc.num_days_from_ce() - 719163;
                    result.push(Some(days_since_epoch as i32));
                }
                Ok(Arc::new(Date32Array::from(result)) as ArrayRef)
            }
            DataType::Timestamp(TimeUnit::Nanosecond, None) => {
                let timestamps_nanos = args[1]
                    .as_any()
                    .downcast_ref::<TimestampNanosecondArray>()
                    .expect("Expected a NanosecondTimeStamp array");
                let timestamp_millis = timestamps_nanos
                    .iter()
                    .map(|t: Option<i64>| t.map(|t| t / 1_000_000))
                    .collect::<TimestampMillisecondArray>();

                let mut result = Vec::new();
                for i in 0..timestamp_millis.len() {
                    if timestamp_millis.is_null(i) {
                        result.push(None);
                        continue;
                    }

                    let timestamp_value = timestamp_millis.value(i);
                    let naive_date_time =
                        NaiveDateTime::from_timestamp_millis(timestamp_value).unwrap();

                    let trunc = match unit.value(i).to_string().as_str() {
                        "second" => naive_date_time
                            .date()
                            .and_hms_opt(
                                naive_date_time.hour(),
                                naive_date_time.minute(),
                                naive_date_time.second(),
                            )
                            .unwrap(),
                        "minute" => naive_date_time
                            .date()
                            .and_hms_opt(
                                naive_date_time.hour(),
                                naive_date_time.minute(),
                                0,
                            )
                            .unwrap(),
                        "hour" => naive_date_time
                            .date()
                            .and_hms_opt(naive_date_time.hour(), 0, 0)
                            .unwrap(),
                        "day" => naive_date_time.date().and_hms_opt(0, 0, 0).unwrap(),
                        "month" => NaiveDate::from_ymd_opt(
                            naive_date_time.year(),
                            naive_date_time.month(),
                            1,
                        )
                        .unwrap()
                        .and_hms_opt(0, 0, 0)
                        .unwrap(),
                        "year" => NaiveDate::from_ymd_opt(naive_date_time.year(), 1, 1)
                            .unwrap()
                            .and_hms_opt(0, 0, 0)
                            .unwrap(),
                        "week" => {
                            let weekday =
                                naive_date_time.weekday().num_days_from_monday() as i64;
                            (naive_date_time - Duration::days(weekday))
                                .date()
                                .and_hms_opt(0, 0, 0)
                                .unwrap()
                        }
                        "quarter" => {
                            let month = (((naive_date_time.month() - 1) / 3) * 3) + 1;
                            NaiveDate::from_ymd_opt(naive_date_time.year(), month, 1)
                                .unwrap()
                                .and_hms_opt(0, 0, 0)
                                .unwrap()
                        }
                        _ => naive_date_time,
                    };

                    let truncated_timestamp = trunc.timestamp_millis();
                    result.push(Some(truncated_timestamp));
                }

                Ok(Arc::new(TimestampMillisecondArray::from(result)) as Arc<dyn Array>)
            }
            DataType::Time64(TimeUnit::Nanosecond) => {
                let times = args[1]
                    .as_any()
                    .downcast_ref::<Time64NanosecondArray>()
                    .expect("Expected a NanosecondTime array");

                let mut result = Vec::new();
                for i in 0..times.len() {
                    if times.is_null(i) {
                        result.push(None);
                        continue;
                    }

                    let time_value = times.value(i); // nanoseconds since midnight
                    let seconds = time_value / 1_000_000_000;
                    let nanoseconds = (time_value % 1_000_000_000) as u32;

                    let naive_time = NaiveTime::from_num_seconds_from_midnight_opt(
                        seconds as u32,
                        nanoseconds,
                    )
                    .unwrap();

                    let trunc = match unit.value(i).to_string().as_str() {
                        "second" => NaiveTime::from_hms_milli_opt(
                            naive_time.hour(),
                            naive_time.minute(),
                            naive_time.second(),
                            0,
                        )
                        .unwrap(),
                        "minute" => NaiveTime::from_hms_milli_opt(
                            naive_time.hour(),
                            naive_time.minute(),
                            0,
                            0,
                        )
                        .unwrap(),
                        "hour" => {
                            NaiveTime::from_hms_milli_opt(naive_time.hour(), 0, 0, 0)
                                .unwrap()
                        }
                        _ => naive_time,
                    };

                    let truncated_nanos = trunc.num_seconds_from_midnight() as i64
                        * 1_000_000_000
                        + trunc.nanosecond() as i64;
                    result.push(Some(truncated_nanos));
                }

                Ok(Arc::new(Time64NanosecondArray::from(result)) as ArrayRef)
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
pub struct DateDiffFunction;

impl ScalarFunctionDef for DateDiffFunction {
    fn name(&self) -> &str {
        "date_diff"
    }

    fn signature(&self) -> Signature {
        Signature::exact(
            vec![
                DataType::Utf8,
                DataType::Timestamp(TimeUnit::Nanosecond, None),
                DataType::Timestamp(TimeUnit::Nanosecond, None),
            ],
            Volatility::Immutable,
        )
    }
    fn return_type(&self) -> ReturnTypeFunction {
        Arc::new(move |_| Ok(Arc::new(DataType::Int64)))
    }

    fn execute(&self, args: &[ArrayRef]) -> Result<ArrayRef> {
        assert_eq!(args.len(), 3);
        let unit_array = args[0]
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("unit should be string");

        let timestamp1_array = args[1]
            .as_any()
            .downcast_ref::<TimestampNanosecondArray>()
            .expect("timestamp1 should be a TimestampNanosecondArray");

        let timestamp2_array = args[2]
            .as_any()
            .downcast_ref::<TimestampNanosecondArray>()
            .expect("timestamp2 should be a TimestampNanosecondArray");

        let array: Vec<i64> = unit_array
            .iter()
            .zip(timestamp1_array.iter())
            .zip(timestamp2_array.iter())
            .map(|((unit_opt, timestamp1_opt), timestamp2_opt)| {
                if let (Some(unit), Some(timestamp1), Some(timestamp2)) =
                    (unit_opt, timestamp1_opt, timestamp2_opt)
                {
                    let dt1 = NaiveDateTime::from_timestamp_opt(
                        timestamp1 / 1_000_000_000,
                        (timestamp1 % 1_000_000_000) as u32,
                    )
                    .unwrap();
                    let dt2 = NaiveDateTime::from_timestamp_opt(
                        timestamp2 / 1_000_000_000,
                        (timestamp2 % 1_000_000_000) as u32,
                    )
                    .unwrap();
                    match unit {
                        "millisecond" => (timestamp2 - timestamp1) / 1_000_000,
                        "second" => (timestamp2 - timestamp1) / 1_000_000_000,
                        "minute" => (timestamp2 - timestamp1) / 60_000_000_000,
                        "hour" => (timestamp2 - timestamp1) / 3_600_000_000_000,
                        "day" => (timestamp2 - timestamp1) / (24 * 3_600_000_000_000),
                        "week" => {
                            (timestamp2 - timestamp1) / (7 * 24 * 3_600_000_000_000)
                        }
                        "month" => {
                            let years = (dt2.year() - dt1.year()) as i64;
                            let months = dt2.month() as i64 - dt1.month() as i64;
                            years * 12 + months
                        }
                        "quarter" => {
                            let years = (dt2.year() - dt1.year()) as i64;
                            let months = dt2.month() as i64 - dt1.month() as i64;
                            (years * 12 + months) / 3
                        }
                        "year" => (dt2.year() - dt1.year()) as i64,
                        _ => 0,
                    }
                } else {
                    0
                }
            })
            .collect();
        Ok(Arc::new(Int64Array::from(array)) as ArrayRef)
    }
}

#[derive(Debug)]
pub struct ParseDurationFunction;

impl ScalarFunctionDef for ParseDurationFunction {
    fn name(&self) -> &str {
        "parse_duration"
    }

    fn signature(&self) -> Signature {
        Signature::exact(vec![DataType::Utf8], Volatility::Immutable)
    }

    fn return_type(&self) -> ReturnTypeFunction {
        Arc::new(move |_| Ok(Arc::new(DataType::Duration(TimeUnit::Nanosecond))))
    }

    fn execute(&self, args: &[ArrayRef]) -> Result<ArrayRef> {
        assert_eq!(args.len(), 1);
        let duration_array = args[0]
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("duration should be string");

        let array: Vec<String> = duration_array
            .iter()
            .map(|duration_opt| {
                if let Some(duration_str) = duration_opt {
                    let re = Regex::new(r"(?i)(\d+(\.\d+)?)\s*([a-z]+)").unwrap();
                    if let Some(caps) = re.captures(duration_str) {
                        if let (Ok(value), Some(unit)) =
                            (caps[1].parse::<f64>(), caps.get(3))
                        {
                            let nanos_total = match unit.as_str().to_lowercase().as_str()
                            {
                                "ns" => value * 1.0,
                                "us" => value * 1_000.0,
                                "ms" => value * 1_000_000.0,
                                "s" => value * 1_000_000_000.0,
                                "m" => value * 60_000_000_000.0,
                                "h" => value * 3_600_000_000_000.0,
                                "d" => value * 86_400_000_000_000.0,
                                _ => 0.0,
                            } as i64;

                            let millis_total =
                                (nanos_total as f64 / 1_000_000.0).round() as i64;
                            let seconds_total = millis_total / 1000;
                            let millis = millis_total % 1000;

                            let days = seconds_total / 86_400;
                            let seconds_remainder = seconds_total % 86_400;

                            let hours = seconds_remainder / 3600;
                            let minutes = (seconds_remainder % 3600) / 60;
                            let seconds = seconds_remainder % 60;

                            format!(
                                "{} {:02}:{:02}:{:02}.{:03}",
                                days, hours, minutes, seconds, millis
                            )
                        } else {
                            "Error: Parsing value".to_string()
                        }
                    } else {
                        "Error: Pattern not matched".to_string()
                    }
                } else {
                    "Error: Null or wrong data".to_string()
                }
            })
            .collect();

        Ok(Arc::new(StringArray::from(array)) as ArrayRef)
    }
}
#[derive(Debug)]
pub struct DateAddFunction;

impl ScalarFunctionDef for DateAddFunction {
    fn name(&self) -> &str {
        "date_add"
    }

    fn signature(&self) -> Signature {
        Signature::one_of(
            vec![
                TypeSignature::Exact(vec![
                    DataType::Utf8,
                    DataType::Int64,
                    DataType::Date32,
                ]),
                TypeSignature::Exact(vec![
                    DataType::Utf8,
                    DataType::Int64,
                    DataType::Time64(TimeUnit::Nanosecond),
                ]),
                TypeSignature::Exact(vec![
                    DataType::Utf8,
                    DataType::Int64,
                    DataType::Timestamp(TimeUnit::Nanosecond, None),
                ]),
            ],
            Volatility::Immutable,
        )
    }

    fn return_type(&self) -> ReturnTypeFunction {
        Arc::new(move |args| Ok(Arc::new(args[2].clone())))
    }

    fn execute(&self, args: &[ArrayRef]) -> Result<ArrayRef> {
        // Ensuring the correct number of arguments
        assert_eq!(args.len(), 3);

        // Extract and validate inputs
        let unit_array = args[0].as_any().downcast_ref::<StringArray>().unwrap();
        let value_array = args[1].as_any().downcast_ref::<Int64Array>().unwrap();

        // // Determine duration to add based on the unit
        let unit = unit_array.value(0);
        let value = value_array.value(0);
        // let duration = parse_duration(unit, value)?;
        match args[2].data_type() {
            DataType::Date32 => {
                let date_array = args[2].as_any().downcast_ref::<Date32Array>().unwrap();
                let mut builder = Date32Array::builder(date_array.len());

                for i in 0..date_array.len() {
                    let date_value = date_array.value(i);
                    // Convert Date32 to NaiveDate
                    let naive_date = chrono::NaiveDate::from_ymd_opt(1970, 1, 1).unwrap()
                        + chrono::Duration::days(date_value as i64);
                    let time = chrono::NaiveTime::from_hms_opt(0, 0, 0).unwrap(); // Assuming time is 00:00:00 for Date32
                    if let Ok(new_date) = parse_duration(naive_date, time, unit, value) {
                        // Convert back from NaiveDate to Date32
                        let duration_since_epoch = new_date.signed_duration_since(
                            chrono::NaiveDate::from_ymd_opt(1970, 1, 1)
                                .unwrap()
                                .and_hms_opt(0, 0, 0)
                                .unwrap(),
                        );
                        builder.append_value(duration_since_epoch.num_days() as i32);
                    } else {
                        builder.append_null();
                    }
                }
                Ok(Arc::new(builder.finish()) as ArrayRef)
            }
            DataType::Time64(TimeUnit::Nanosecond) => {
                let time_array = args[2]
                    .as_any()
                    .downcast_ref::<Time64NanosecondArray>()
                    .unwrap();
                let mut builder = Time64NanosecondArray::builder(time_array.len());

                for i in 0..time_array.len() {
                    let time_value = time_array.value(i);
                    let seconds = (time_value / 1_000_000_000) as u32; // Get seconds part
                    let nanosecond = (time_value % 1_000_000_000) as u32; // Get nanoseconds part
                    let time = chrono::NaiveTime::from_num_seconds_from_midnight_opt(
                        seconds, nanosecond,
                    )
                    .unwrap();

                    let date = chrono::NaiveDate::from_ymd_opt(1970, 1, 1).unwrap(); // Assuming date part is the epoch start for Time32
                    if let Ok(new_time) = parse_duration(date, time, unit, value as i64) {
                        // Convert back from NaiveTime to Time64Nanosecond
                        let duration_since_midnight =
                            (new_time.num_seconds_from_midnight() as i64 * 1_000_000_000)
                                + new_time.nanosecond() as i64;
                        builder.append_value(duration_since_midnight);
                    } else {
                        builder.append_null();
                    }
                }
                Ok(Arc::new(builder.finish()) as ArrayRef)
            }
            DataType::Timestamp(TimeUnit::Nanosecond, _) => {
                let timestamp_array = args[2]
                    .as_any()
                    .downcast_ref::<TimestampNanosecondArray>()
                    .unwrap();
                let timestamp_milli_arr = timestamp_array
                    .iter()
                    .map(|timestamp| timestamp.map(|timestamp| timestamp / 1_000_000))
                    .collect::<TimestampMillisecondArray>();

                let mut builder =
                    TimestampMillisecondArray::builder(timestamp_milli_arr.len());

                for i in 0..timestamp_milli_arr.len() {
                    let timestamp_value = timestamp_milli_arr.value(i);
                    // Convert Timestamp to NaiveDateTime
                    let datetime =
                        chrono::NaiveDateTime::from_timestamp_millis(timestamp_value)
                            .unwrap();
                    if let Ok(new_datetime) =
                        parse_duration(datetime.date(), datetime.time(), unit, value)
                    {
                        // Convert back from NaiveDateTime to Timestamp
                        builder.append_value(new_datetime.timestamp_millis());
                    } else {
                        builder.append_null();
                    }
                }
                Ok(Arc::new(builder.finish()) as ArrayRef)
            }

            _ => Err(ArrowError::InvalidArgumentError(
                "Invalid input type".to_string(),
            ))
            .map_err(|err| DataFusionError::Execution(format!("Cast error: {}", err))),
        }
    }
}
fn parse_duration(
    date: chrono::NaiveDate,
    time: chrono::NaiveTime,
    unit: &str,
    value: i64,
) -> Result<chrono::NaiveDateTime, DataFusionError> {
    match unit {
        "millisecond" => Ok(chrono::NaiveDateTime::new(date, time)
            + chrono::Duration::milliseconds(value as i64)),
        "second" => Ok(chrono::NaiveDateTime::new(date, time)
            + chrono::Duration::seconds((value) as i64)),
        "minute" => Ok(chrono::NaiveDateTime::new(date, time)
            + chrono::Duration::minutes(value as i64)),
        "hour" => Ok(chrono::NaiveDateTime::new(date, time)
            + chrono::Duration::hours(value as i64)),
        "day" => {
            Ok(chrono::NaiveDateTime::new(date, time)
                + chrono::Duration::days(value as i64))
        }
        "week" => Ok(chrono::NaiveDateTime::new(date, time)
            + chrono::Duration::weeks(value as i64)),
        "month" => {
            let new_date = if value >= 0 {
                date.checked_add_months(Months::new(value as u32))
            } else {
                date.checked_sub_months(Months::new(value.abs() as u32))
            };
            new_date
                .map(|d| chrono::NaiveDateTime::new(d, time))
                .ok_or_else(|| {
                    DataFusionError::Execution(format!("Error adding months: {}", value))
                })
        }
        "quarter" => {
            let new_date = if value >= 0 {
                date.checked_add_months(Months::new((value * 3) as u32))
            } else {
                date.checked_sub_months(Months::new((value * 3).abs() as u32))
            };
            new_date
                .map(|d| chrono::NaiveDateTime::new(d, time))
                .ok_or_else(|| {
                    DataFusionError::Execution(format!("Error adding months: {}", value))
                })
        }
        "year" => {
            let new_date = if value >= 0 {
                date.checked_add_months(Months::new((value * 12) as u32))
            } else {
                date.checked_sub_months(Months::new((value * 12).abs() as u32))
            };
            new_date
                .map(|d| chrono::NaiveDateTime::new(d, time))
                .ok_or_else(|| {
                    DataFusionError::Execution(format!("Error adding months: {}", value))
                })
        }
        _ => Err(DataFusionError::Execution(format!(
            "Invalid unit for duration: {}",
            unit
        ))),
    }
}
#[derive(Debug)]
pub struct LastDayOfMonthFunction;

impl ScalarFunctionDef for LastDayOfMonthFunction {
    fn name(&self) -> &str {
        "last_day_of_month"
    }

    fn signature(&self) -> Signature {
        Signature::one_of(
            vec![
                TypeSignature::Exact(vec![DataType::Date32]),
                TypeSignature::Exact(vec![DataType::Timestamp(
                    TimeUnit::Nanosecond,
                    None,
                )]),
            ],
            Volatility::Immutable,
        )
    }

    fn return_type(&self) -> ReturnTypeFunction {
        let return_type = Arc::new(DataType::Date32);
        Arc::new(move |_| Ok(return_type.clone()))
    }

    fn execute(&self, args: &[ArrayRef]) -> Result<ArrayRef> {
        if args.is_empty() || args.len() > 1 {
            return Err(datafusion::error::DataFusionError::Execution(
                "last_day_of_month function takes exactly one argument".to_string(),
            ));
        }
        let result = match args[0].data_type() {
            DataType::Date32 => {
                let input = args[0]
                    .as_any()
                    .downcast_ref::<Date32Array>()
                    .expect("Failed to downcast to Date32Array");

                let mut res = Vec::new();
                for i in 0..input.len() {
                    let date_value = input.value(i);
                    let naive_date =
                        NaiveDate::from_num_days_from_ce_opt(date_value as i32).unwrap();
                    let last_day = cal_last_day_of_month(naive_date);
                    let epoch = NaiveDate::from_ymd_opt(1, 1, 1).unwrap();
                    let days_since_epoch =
                        last_day.signed_duration_since(epoch).num_days();
                    res.push(Some(days_since_epoch as i32));
                }
                Arc::new(Date32Array::from(res)) as ArrayRef
            }
            DataType::Timestamp(TimeUnit::Nanosecond, None) => {
                let timestamp_array = args[0]
                    .as_any()
                    .downcast_ref::<TimestampNanosecondArray>()
                    .expect("Expected a NanosecondTimeStamp array");
                let milliseconds_array = timestamp_array
                    .iter()
                    .map(|timestamp| timestamp.map(|timestamp| timestamp / 1_000_000))
                    .collect::<TimestampMillisecondArray>();

                let mut res = Vec::new();
                for i in 0..milliseconds_array.len() {
                    let timestamp_value = milliseconds_array.value(i);
                    let naive_datetime =
                        Utc.timestamp_millis_opt(timestamp_value).unwrap();
                    let last_day = cal_last_day_of_month(naive_datetime.date_naive());
                    let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).unwrap();
                    let days_since_epoch =
                        last_day.signed_duration_since(epoch).num_days();
                    res.push(Some(days_since_epoch as i32));
                }
                Arc::new(Date32Array::from(res)) as ArrayRef
            }
            //timestamp todo
            _ => {
                return Err(datafusion::error::DataFusionError::Execution(
                    "Invalid input type for last_day_of_month function".to_string(),
                ))
            }
        };
        Ok(result)
    }
}

#[derive(Debug)]
pub struct DayOfWeekFunction;

impl ScalarFunctionDef for DayOfWeekFunction {
    fn name(&self) -> &str {
        "day_of_week"
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
        let return_type = Arc::new(DataType::Int64); // Returning as bigint
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

                let weekdays: Vec<i64> = date_array
                    .iter()
                    .map(|date_opt| {
                        date_opt
                            .map(|date| {
                                let naive_date = Date32Type::to_naive_date(date);
                                // Convert to ISO weekday
                                naive_date.weekday().number_from_monday() as i64
                            })
                            .unwrap_or(0)
                    })
                    .collect();

                Ok(Arc::new(Int64Array::from(weekdays)) as ArrayRef)
            }
            DataType::Timestamp(TimeUnit::Nanosecond, _) => {
                let timestamp_array = input
                    .as_any()
                    .downcast_ref::<TimestampNanosecondArray>()
                    .expect("Expected a nanosecond timestamp array");

                let weekdays: Vec<i64> = timestamp_array
                    .iter()
                    .map(|timestamp_opt| {
                        timestamp_opt
                            .map(|timestamp| {
                                let datetime = NaiveDateTime::from_timestamp_opt(
                                    timestamp / 1_000_000_000,
                                    (timestamp % 1_000_000_000) as u32,
                                );
                                datetime
                                    .map(|dt| dt.weekday().number_from_monday() as i64)
                                    .unwrap_or(0)
                            })
                            .unwrap_or(0)
                    })
                    .collect();

                Ok(Arc::new(Int64Array::from(weekdays)) as ArrayRef)
            }
            _ => Err(ArrowError::InvalidArgumentError(
                "Invalid input type".to_string(),
            )),
        };
        Ok(result?)
    }
}

#[derive(Debug)]
pub struct DayOfYearFunction;

impl ScalarFunctionDef for DayOfYearFunction {
    fn name(&self) -> &str {
        "day_of_year"
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
        let return_type = Arc::new(DataType::Int64); // Returning as bigint
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

                let day_of_years: Vec<i64> = date_array
                    .iter()
                    .map(|date_opt| {
                        date_opt
                            .map(|date| {
                                let naive_date = Date32Type::to_naive_date(date);
                                naive_date.ordinal() as i64 // Gets the day of the year
                            })
                            .unwrap_or(0)
                    })
                    .collect();

                Ok(Arc::new(Int64Array::from(day_of_years)) as ArrayRef)
            }
            DataType::Timestamp(TimeUnit::Nanosecond, _) => {
                let timestamp_array = input
                    .as_any()
                    .downcast_ref::<TimestampNanosecondArray>()
                    .expect("Expected a nanosecond timestamp array");

                let day_of_years: Vec<i64> = timestamp_array
                    .iter()
                    .map(|timestamp_opt| {
                        timestamp_opt
                            .map(|timestamp| {
                                let datetime = NaiveDateTime::from_timestamp_opt(
                                    timestamp / 1_000_000_000,
                                    (timestamp % 1_000_000_000) as u32,
                                );
                                datetime.map(|dt| dt.ordinal() as i64).unwrap_or(0)
                                // day of the year
                            })
                            .unwrap_or(0)
                    })
                    .collect();

                Ok(Arc::new(Int64Array::from(day_of_years)) as ArrayRef)
            }
            _ => Err(ArrowError::InvalidArgumentError(
                "Invalid input type".to_string(),
            )),
        };
        Ok(result?)
    }
}
#[derive(Debug)]
pub struct HourFunction;

impl ScalarFunctionDef for HourFunction {
    fn name(&self) -> &str {
        "hour"
    }

    fn signature(&self) -> Signature {
        // Function accepts Date32 or Timestamp
        Signature::one_of(
            vec![
                TypeSignature::Exact(vec![DataType::Date32]),
                TypeSignature::Exact(vec![DataType::Timestamp(
                    TimeUnit::Nanosecond,
                    None,
                )]),
            ],
            Volatility::Immutable,
        )
    }

    fn return_type(&self) -> ReturnTypeFunction {
        let return_type = Arc::new(DataType::Int64); // Returning as bigint
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

                // Date32 doesn't inherently contain hour information, defaulting all to 0
                let hours: Vec<i64> = date_array
                    .iter()
                    .map(|_| 0) // Every date defaults to hour 0
                    .collect();

                Ok(Arc::new(Int64Array::from(hours)) as ArrayRef)
            }
            DataType::Timestamp(TimeUnit::Nanosecond, _) => {
                let timestamp_array = input
                    .as_any()
                    .downcast_ref::<TimestampNanosecondArray>()
                    .expect("Expected a nanosecond timestamp array");

                let hours: Vec<i64> = timestamp_array
                    .iter()
                    .map(|timestamp_opt| {
                        timestamp_opt
                            .map(|timestamp| {
                                let datetime = NaiveDateTime::from_timestamp_opt(
                                    timestamp / 1_000_000_000,
                                    (timestamp % 1_000_000_000) as u32,
                                );
                                datetime.map(|dt| dt.hour() as i64).unwrap_or(0)
                            })
                            .unwrap_or(0)
                    })
                    .collect();

                Ok(Arc::new(Int64Array::from(hours)) as ArrayRef)
            }
            _ => Err(ArrowError::InvalidArgumentError(
                "Invalid input type".to_string(),
            )),
        };
        Ok(result?)
    }
}

#[derive(Debug)]
pub struct MillisecondFunction;

impl ScalarFunctionDef for MillisecondFunction {
    fn name(&self) -> &str {
        "millisecond"
    }

    fn signature(&self) -> Signature {
        // Function accepts Timestamp
        Signature::one_of(
            vec![TypeSignature::Exact(vec![DataType::Timestamp(
                TimeUnit::Nanosecond,
                None,
            )])],
            Volatility::Immutable,
        )
    }

    fn return_type(&self) -> ReturnTypeFunction {
        let return_type = Arc::new(DataType::Int64); // Returning as bigint
        Arc::new(move |_| Ok(return_type.clone()))
    }

    fn execute(&self, args: &[ArrayRef]) -> Result<ArrayRef> {
        assert_eq!(args.len(), 1);

        let input = &args[0];
        let result = match input.data_type() {
            DataType::Timestamp(TimeUnit::Nanosecond, _) => {
                let timestamp_array = input
                    .as_any()
                    .downcast_ref::<TimestampNanosecondArray>()
                    .expect("Expected a nanosecond timestamp array");

                let milliseconds: Vec<i64> = timestamp_array
                    .iter()
                    .map(|timestamp_opt| {
                        timestamp_opt
                            .map(|timestamp| (timestamp % 1_000_000_000) / 1_000_000) // extract milliseconds from nanoseconds
                            .unwrap_or(0)
                    })
                    .collect();

                Ok(Arc::new(Int64Array::from(milliseconds)) as ArrayRef)
            }
            _ => Err(ArrowError::InvalidArgumentError(
                "Invalid input type".to_string(),
            )),
        };
        Ok(result?)
    }
}

#[derive(Debug)]
pub struct MinuteFunction;

impl ScalarFunctionDef for MinuteFunction {
    fn name(&self) -> &str {
        "minute"
    }

    fn signature(&self) -> Signature {
        // Function accepts Timestamp
        Signature::one_of(
            vec![TypeSignature::Exact(vec![DataType::Timestamp(
                TimeUnit::Nanosecond,
                None,
            )])],
            Volatility::Immutable,
        )
    }

    fn return_type(&self) -> ReturnTypeFunction {
        let return_type = Arc::new(DataType::Int64); // Returning as bigint
        Arc::new(move |_| Ok(return_type.clone()))
    }

    fn execute(&self, args: &[ArrayRef]) -> Result<ArrayRef> {
        assert_eq!(args.len(), 1);

        let input = &args[0];
        let result = match input.data_type() {
            DataType::Timestamp(TimeUnit::Nanosecond, _) => {
                let timestamp_array = input
                    .as_any()
                    .downcast_ref::<TimestampNanosecondArray>()
                    .expect("Expected a nanosecond timestamp array");

                let minutes: Vec<i64> = timestamp_array
                    .iter()
                    .map(|timestamp_opt| {
                        timestamp_opt
                            .map(|timestamp| {
                                let datetime = NaiveDateTime::from_timestamp_opt(
                                    timestamp / 1_000_000_000,
                                    (timestamp % 1_000_000_000) as u32,
                                );
                                datetime.map(|dt| dt.minute() as i64).unwrap_or(0)
                            })
                            .unwrap_or(0)
                    })
                    .collect();

                Ok(Arc::new(Int64Array::from(minutes)) as ArrayRef)
            }
            _ => Err(ArrowError::InvalidArgumentError(
                "Invalid input type".to_string(),
            )),
        };
        Ok(result?)
    }
}

#[derive(Debug)]
pub struct MonthFunction;

impl ScalarFunctionDef for MonthFunction {
    fn name(&self) -> &str {
        "month"
    }

    fn signature(&self) -> Signature {
        // Function accepts Date32 and Timestamp
        Signature::one_of(
            vec![
                TypeSignature::Exact(vec![DataType::Date32]),
                TypeSignature::Exact(vec![DataType::Timestamp(
                    TimeUnit::Nanosecond,
                    None,
                )]),
            ],
            Volatility::Immutable,
        )
    }

    fn return_type(&self) -> ReturnTypeFunction {
        let return_type = Arc::new(DataType::Int64); // Returning as bigint
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

                let months: Vec<i64> = date_array
                    .iter()
                    .map(|date_opt| {
                        date_opt
                            .map(|date| {
                                let naive_date = Date32Type::to_naive_date(date);
                                naive_date.month() as i64
                            })
                            .unwrap_or(0)
                    })
                    .collect();

                Ok(Arc::new(Int64Array::from(months)) as ArrayRef)
            }
            DataType::Timestamp(TimeUnit::Nanosecond, _) => {
                let timestamp_array = input
                    .as_any()
                    .downcast_ref::<TimestampNanosecondArray>()
                    .expect("Expected a nanosecond timestamp array");

                let months: Vec<i64> = timestamp_array
                    .iter()
                    .map(|timestamp_opt| {
                        timestamp_opt
                            .map(|timestamp| {
                                let datetime = NaiveDateTime::from_timestamp_opt(
                                    timestamp / 1_000_000_000,
                                    (timestamp % 1_000_000_000) as u32,
                                );
                                datetime.map(|dt| dt.month() as i64).unwrap_or(0)
                            })
                            .unwrap_or(0)
                    })
                    .collect();

                Ok(Arc::new(Int64Array::from(months)) as ArrayRef)
            }
            _ => Err(ArrowError::InvalidArgumentError(
                "Invalid input type".to_string(),
            )),
        };
        Ok(result?)
    }
}

#[derive(Debug)]
pub struct QuarterFunction;

impl ScalarFunctionDef for QuarterFunction {
    fn name(&self) -> &str {
        "quarter"
    }

    fn signature(&self) -> Signature {
        // Function accepts Date32 and Timestamp
        Signature::one_of(
            vec![
                TypeSignature::Exact(vec![DataType::Date32]),
                TypeSignature::Exact(vec![DataType::Timestamp(
                    TimeUnit::Nanosecond,
                    None,
                )]),
            ],
            Volatility::Immutable,
        )
    }

    fn return_type(&self) -> ReturnTypeFunction {
        let return_type = Arc::new(DataType::Int64); // Returning as bigint
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

                let quarters: Vec<i64> = date_array
                    .iter()
                    .map(|date_opt| {
                        date_opt
                            .map(|date| {
                                let naive_date = Date32Type::to_naive_date(date);
                                ((naive_date.month() - 1) / 3 + 1) as i64
                            })
                            .unwrap_or(0)
                    })
                    .collect();

                Ok(Arc::new(Int64Array::from(quarters)) as ArrayRef)
            }
            DataType::Timestamp(TimeUnit::Nanosecond, _) => {
                let timestamp_array = input
                    .as_any()
                    .downcast_ref::<TimestampNanosecondArray>()
                    .expect("Expected a nanosecond timestamp array");

                let quarters: Vec<i64> = timestamp_array
                    .iter()
                    .map(|timestamp_opt| {
                        timestamp_opt
                            .map(|timestamp| {
                                let datetime = NaiveDateTime::from_timestamp_opt(
                                    timestamp / 1_000_000_000,
                                    (timestamp % 1_000_000_000) as u32,
                                );
                                datetime
                                    .map(|dt| ((dt.month() - 1) / 3 + 1) as i64) // calculate quarter
                                    .unwrap_or(0)
                            })
                            .unwrap_or(0)
                    })
                    .collect();

                Ok(Arc::new(Int64Array::from(quarters)) as ArrayRef)
            }
            _ => Err(ArrowError::InvalidArgumentError(
                "Invalid input type".to_string(),
            )),
        };
        Ok(result?)
    }
}

#[derive(Debug)]
pub struct SecondFunction;

impl ScalarFunctionDef for SecondFunction {
    fn name(&self) -> &str {
        "second"
    }

    fn signature(&self) -> Signature {
        // Function accepts Date32 and Timestamp
        Signature::one_of(
            vec![
                TypeSignature::Exact(vec![DataType::Date32]),
                TypeSignature::Exact(vec![DataType::Timestamp(
                    TimeUnit::Nanosecond,
                    None,
                )]),
            ],
            Volatility::Immutable,
        )
    }

    fn return_type(&self) -> ReturnTypeFunction {
        let return_type = Arc::new(DataType::Int64); // Returning as bigint
        Arc::new(move |_| Ok(return_type.clone()))
    }

    fn execute(&self, args: &[ArrayRef]) -> Result<ArrayRef> {
        assert_eq!(args.len(), 1);

        let input = &args[0];
        let result = match input.data_type() {
            DataType::Date32 => {
                // For Date32, return 0 for all entries as there is no specific second information
                let date_array = input
                    .as_any()
                    .downcast_ref::<Date32Array>()
                    .expect("Expected a date array");
                let seconds: Vec<i64> = date_array.iter().map(|_| 0).collect();
                Ok(Arc::new(Int64Array::from(seconds)) as ArrayRef)
            }
            DataType::Timestamp(TimeUnit::Nanosecond, _) => {
                let timestamp_array = input
                    .as_any()
                    .downcast_ref::<TimestampNanosecondArray>()
                    .expect("Expected a nanosecond timestamp array");

                let seconds: Vec<i64> = timestamp_array
                    .iter()
                    .map(|timestamp_opt| {
                        timestamp_opt
                            .map(|timestamp| {
                                let datetime = NaiveDateTime::from_timestamp_opt(
                                    timestamp / 1_000_000_000,
                                    (timestamp % 1_000_000_000) as u32,
                                );
                                datetime.map(|dt| dt.second() as i64).unwrap_or(0)
                            })
                            .unwrap_or(0)
                    })
                    .collect();

                Ok(Arc::new(Int64Array::from(seconds)) as ArrayRef)
            }
            _ => Err(ArrowError::InvalidArgumentError(
                "Invalid input type".to_string(),
            )),
        };
        Ok(result?)
    }
}

#[derive(Debug)]
pub struct WeekFunction;

impl ScalarFunctionDef for WeekFunction {
    fn name(&self) -> &str {
        "week"
    }

    fn signature(&self) -> Signature {
        // Function accepts Date32 and Timestamp
        Signature::one_of(
            vec![
                TypeSignature::Exact(vec![DataType::Date32]),
                TypeSignature::Exact(vec![DataType::Timestamp(
                    TimeUnit::Nanosecond,
                    None,
                )]),
            ],
            Volatility::Immutable,
        )
    }

    fn return_type(&self) -> ReturnTypeFunction {
        let return_type = Arc::new(DataType::Int64); // Returning as bigint
        Arc::new(move |_| Ok(return_type.clone()))
    }

    fn execute(&self, args: &[ArrayRef]) -> Result<ArrayRef> {
        assert_eq!(args.len(), 1);

        let input = &args[0];
        let result =
            match input.data_type() {
                DataType::Date32 | DataType::Timestamp(TimeUnit::Nanosecond, _) => {
                    let date_iter: Box<dyn Iterator<Item = Option<NaiveDate>>> =
                        match input.data_type() {
                            DataType::Date32 => {
                                let date_array = input
                                    .as_any()
                                    .downcast_ref::<Date32Array>()
                                    .expect("Expected a date array");
                                Box::new(date_array.iter().map(|date_opt| {
                                    date_opt.map(Date32Type::to_naive_date)
                                }))
                            }
                            DataType::Timestamp(TimeUnit::Nanosecond, _) => {
                                let timestamp_array = input
                                    .as_any()
                                    .downcast_ref::<TimestampNanosecondArray>()
                                    .expect("Expected a nanosecond timestamp array");
                                Box::new(timestamp_array.iter().map(|timestamp_opt| {
                                    timestamp_opt
                                        .map(|timestamp| {
                                            let seconds = timestamp / 1_000_000_000;
                                            let ns = (timestamp % 1_000_000_000) as u32;
                                            NaiveDateTime::from_timestamp_opt(seconds, ns)
                                                .map(|dt| dt.date())
                                        })
                                        .flatten()
                                }))
                            }
                            _ => unreachable!(),
                        };

                    let weeks: Vec<i64> = date_iter
                        .map(|date_opt| {
                            date_opt
                                .map(|date| date.iso_week().week() as i64)
                                .unwrap_or(0)
                        })
                        .collect();

                    Ok(Arc::new(Int64Array::from(weeks)) as ArrayRef)
                }
                _ => Err(ArrowError::InvalidArgumentError(
                    "Invalid input type".to_string(),
                )),
            };
        Ok(result?)
    }
}

#[derive(Debug)]
pub struct YearFunction;

impl ScalarFunctionDef for YearFunction {
    fn name(&self) -> &str {
        "year"
    }

    fn signature(&self) -> Signature {
        // Function accepts Date32 and Timestamp
        Signature::one_of(
            vec![
                TypeSignature::Exact(vec![DataType::Date32]),
                TypeSignature::Exact(vec![DataType::Timestamp(
                    TimeUnit::Nanosecond,
                    None,
                )]),
            ],
            Volatility::Immutable,
        )
    }

    fn return_type(&self) -> ReturnTypeFunction {
        let return_type = Arc::new(DataType::Int64); // Returning as bigint
        Arc::new(move |_| Ok(return_type.clone()))
    }

    fn execute(&self, args: &[ArrayRef]) -> Result<ArrayRef> {
        assert_eq!(args.len(), 1);

        let input = &args[0];
        let result = match input.data_type() {
            DataType::Date32 | DataType::Timestamp(TimeUnit::Nanosecond, _) => {
                let date_iter: Box<dyn Iterator<Item = Option<NaiveDate>>> =
                    match input.data_type() {
                        DataType::Date32 => {
                            let date_array = input
                                .as_any()
                                .downcast_ref::<Date32Array>()
                                .expect("Expected a date array");
                            Box::new(
                                date_array.iter().map(|date_opt| {
                                    date_opt.map(Date32Type::to_naive_date)
                                }),
                            )
                        }
                        DataType::Timestamp(TimeUnit::Nanosecond, _) => {
                            let timestamp_array = input
                                .as_any()
                                .downcast_ref::<TimestampNanosecondArray>()
                                .expect("Expected a nanosecond timestamp array");
                            Box::new(timestamp_array.iter().map(|timestamp_opt| {
                                timestamp_opt
                                    .map(|timestamp| {
                                        let seconds = timestamp / 1_000_000_000;
                                        let ns = (timestamp % 1_000_000_000) as u32;
                                        NaiveDateTime::from_timestamp_opt(seconds, ns)
                                            .map(|dt| dt.date())
                                    })
                                    .flatten()
                            }))
                        }
                        _ => unreachable!(),
                    };

                let years: Vec<i64> = date_iter
                    .map(|date_opt| date_opt.map(|date| date.year() as i64).unwrap_or(0))
                    .collect();

                Ok(Arc::new(Int64Array::from(years)) as ArrayRef)
            }
            _ => Err(ArrowError::InvalidArgumentError(
                "Invalid input type".to_string(),
            )),
        };
        Ok(result?)
    }
}

#[derive(Debug)]
pub struct YearOfWeekFunction;

impl ScalarFunctionDef for YearOfWeekFunction {
    fn name(&self) -> &str {
        "year_of_week"
    }

    fn signature(&self) -> Signature {
        // Function accepts Date32 and Timestamp
        Signature::one_of(
            vec![
                TypeSignature::Exact(vec![DataType::Date32]),
                TypeSignature::Exact(vec![DataType::Timestamp(
                    TimeUnit::Nanosecond,
                    None,
                )]),
            ],
            Volatility::Immutable,
        )
    }

    fn return_type(&self) -> ReturnTypeFunction {
        let return_type = Arc::new(DataType::Int64); // Returning as bigint
        Arc::new(move |_| Ok(return_type.clone()))
    }

    fn execute(&self, args: &[ArrayRef]) -> Result<ArrayRef> {
        assert_eq!(args.len(), 1);

        let input = &args[0];
        let result =
            match input.data_type() {
                DataType::Date32 | DataType::Timestamp(TimeUnit::Nanosecond, _) => {
                    let date_iter: Box<dyn Iterator<Item = Option<NaiveDate>>> =
                        match input.data_type() {
                            DataType::Date32 => {
                                let date_array = input
                                    .as_any()
                                    .downcast_ref::<Date32Array>()
                                    .expect("Expected a date array");
                                Box::new(date_array.iter().map(|date_opt| {
                                    date_opt.map(Date32Type::to_naive_date)
                                }))
                            }
                            DataType::Timestamp(TimeUnit::Nanosecond, _) => {
                                let timestamp_array = input
                                    .as_any()
                                    .downcast_ref::<TimestampNanosecondArray>()
                                    .expect("Expected a nanosecond timestamp array");
                                Box::new(timestamp_array.iter().map(|timestamp_opt| {
                                    timestamp_opt
                                        .map(|timestamp| {
                                            let seconds = timestamp / 1_000_000_000;
                                            let ns = (timestamp % 1_000_000_000) as u32;
                                            NaiveDateTime::from_timestamp_opt(seconds, ns)
                                                .map(|dt| dt.date())
                                        })
                                        .flatten()
                                }))
                            }
                            _ => unreachable!(),
                        };

                    let year_of_weeks: Vec<i64> = date_iter
                        .map(|date_opt| {
                            date_opt
                                .map(|date| date.iso_week().year() as i64) // Retrieve year of the ISO week
                                .unwrap_or(0)
                        })
                        .collect();

                    Ok(Arc::new(Int64Array::from(year_of_weeks)) as ArrayRef)
                }
                _ => Err(ArrowError::InvalidArgumentError(
                    "Invalid input type".to_string(),
                )),
            };
        Ok(result?)
    }
}

fn cal_last_day_of_month(date: NaiveDate) -> NaiveDate {
    let year = date.year();
    let month = date.month();
    NaiveDate::from_ymd_opt(year, month, 1).unwrap()
        + Duration::days(
            (NaiveDate::from_ymd_opt(year, month.checked_add(1).unwrap_or(1), 1)
                .unwrap()
                - NaiveDate::from_ymd_opt(year, month, 1).unwrap())
            .num_days()
                - 1,
        )
}

enum DateFormatStatus {
    Default,
    PostPercent,
}
#[derive(Debug)]
pub struct DateParseFunction;

impl ScalarFunctionDef for DateParseFunction {
    fn name(&self) -> &str {
        "date_parse"
    }

    fn signature(&self) -> Signature {
        Signature::exact(vec![DataType::Utf8, DataType::Utf8], Volatility::Immutable)
    }

    fn return_type(&self) -> ReturnTypeFunction {
        Arc::new(move |_| Ok(Arc::new(DataType::Timestamp(TimeUnit::Nanosecond, None))))
    }
    fn execute(&self, args: &[ArrayRef]) -> Result<ArrayRef> {
        let string_array = args[0].as_any().downcast_ref::<StringArray>().unwrap();
        let format_array = args[1].as_any().downcast_ref::<StringArray>().unwrap();

        let mut timestamps = Vec::new();

        let mut year = 1970;
        let mut month = 1;
        let mut day = 1;
        let mut hour = 0;
        let mut minute = 0;
        let mut second = 0;
        let mut millis = 0;

        let month_map: HashMap<&str, u32> = [
            // Abbreviated month names
            ("jan", 1),
            ("feb", 2),
            ("mar", 3),
            ("apr", 4),
            ("may", 5),
            ("jun", 6),
            ("jul", 7),
            ("aug", 8),
            ("sep", 9),
            ("oct", 10),
            ("nov", 11),
            ("dec", 12),
            // Full month names
            ("january", 1),
            ("february", 2),
            ("march", 3),
            ("april", 4),
            ("may", 5),
            ("june", 6),
            ("july", 7),
            ("august", 8),
            ("september", 9),
            ("october", 10),
            ("november", 11),
            ("december", 12),
        ]
        .iter()
        .copied()
        .collect();

        let weekday_map: HashMap<&str, chrono::Weekday> = [
            // Abbreviated weekday names
            ("sun", chrono::Weekday::Sun),
            ("mon", chrono::Weekday::Mon),
            ("tue", chrono::Weekday::Tue),
            ("wed", chrono::Weekday::Wed),
            ("thu", chrono::Weekday::Thu),
            ("fri", chrono::Weekday::Fri),
            ("sat", chrono::Weekday::Sat),
            // Full weekday names
            ("sunday", chrono::Weekday::Sun),
            ("monday", chrono::Weekday::Mon),
            ("tuesday", chrono::Weekday::Tue),
            ("wednesday", chrono::Weekday::Wed),
            ("thursday", chrono::Weekday::Thu),
            ("friday", chrono::Weekday::Fri),
            ("saturday", chrono::Weekday::Sat),
        ]
        .iter()
        .cloned()
        .collect();

        for i in 0..string_array.len() {
            let date_str = string_array.value(i);
            let format_str = format_array.value(i);
            let mut status = DateFormatStatus::Default;
            let mut index = 0;
            for c in format_str.chars() {
                match status {
                    DateFormatStatus::Default => match c {
                        '%' => status = DateFormatStatus::PostPercent,
                        c => {
                            if date_str.as_bytes()[index] as char == c {
                                index += 1;
                            } else {
                                return Err(DataFusionError::Execution(format!(
                                    "Mismatch in format and date string at position {}: expected '{}', found '{}'",
                                    index,
                                    c,
                                    date_str.as_bytes()[index] as char
                                )));
                            }
                        }
                    },
                    DateFormatStatus::PostPercent => match c {
                        'Y' => match date_str[index..index + 4].parse::<i32>() {
                            Ok(value) => {
                                year = value;
                                index = index + 4;
                                status = DateFormatStatus::Default;
                            }
                            Err(e) => {
                                return Err(DataFusionError::Execution(format!(
                                    "Unable to parse year: {}",
                                    e
                                )))
                            }
                        },
                        'y' => match date_str[index..index + 2].parse::<i32>() {
                            Ok(value) => {
                                year = if value >= 70 {
                                    1900 + value
                                } else {
                                    2000 + value
                                };
                                index += 2;
                                status = DateFormatStatus::Default;
                            }
                            Err(e) => {
                                return Err(DataFusionError::Execution(format!(
                                    "Unable to parse year: {}",
                                    e
                                )));
                            }
                        },
                        'c' | 'm' => match parse_next_number(&date_str, index) {
                            Ok((parsed_day, end_index)) => {
                                if parsed_day >= 1 && parsed_day <= 12 {
                                    month = parsed_day;
                                    index = end_index;
                                    status = DateFormatStatus::Default;
                                } else {
                                    return Err(DataFusionError::Execution(
                                        "Invalid month".into(),
                                    ));
                                }
                            }
                            Err(e) => return Err(e),
                        },
                        'd' | 'e' => match parse_next_number(&date_str, index) {
                            Ok((value, end_index)) => {
                                if value >= 1 && value <= 31 {
                                    day = value;
                                    index = end_index;
                                    status = DateFormatStatus::Default;
                                } else {
                                    return Err(DataFusionError::Execution(
                                        "Invalid day".into(),
                                    ));
                                }
                            }
                            Err(e) => {
                                return Err(DataFusionError::Execution(format!(
                                    "Unable to parse day: {}",
                                    e
                                )))
                            }
                        },
                        'H' | 'k' => match parse_next_number(&date_str, index) {
                            Ok((value, end_index)) => {
                                if value < 24 {
                                    hour = value;
                                    index = end_index;
                                    status = DateFormatStatus::Default;
                                } else {
                                    return Err(DataFusionError::Execution(
                                        "Invalid 24-hour".into(),
                                    ));
                                }
                            }
                            Err(e) => {
                                return Err(DataFusionError::Execution(format!(
                                    "Unable to parse 24-hour: {}",
                                    e
                                )))
                            }
                        },
                        'h' | 'I' | 'l' => match parse_next_number(&date_str, index) {
                            Ok((value, end_index)) => {
                                if value == 12 {
                                    hour = 0;
                                    index = end_index;
                                    status = DateFormatStatus::Default;
                                } else if value < 12 {
                                    hour = value;
                                    index = end_index;
                                    status = DateFormatStatus::Default;
                                } else {
                                    return Err(DataFusionError::Execution(
                                        "Invalid 12-hour".into(),
                                    ));
                                }
                            }
                            Err(e) => {
                                return Err(DataFusionError::Execution(format!(
                                    "Unable to parse 12-hour: {}",
                                    e
                                )))
                            }
                        },
                        'p' => match date_str[index..index + 2].to_uppercase().as_str() {
                            "AM" => {
                                if hour == 12 {
                                    hour = 0;
                                }
                                index += 2;
                                status = DateFormatStatus::Default;
                            }
                            "PM" => {
                                if hour != 12 {
                                    hour += 12;
                                }
                                index += 2;
                                status = DateFormatStatus::Default;
                            }
                            _ => {
                                return Err(DataFusionError::Execution(
                                    "Invalid AM/PM specifier".into(),
                                ));
                            }
                        },
                        'i' => match date_str[index..index + 2].parse::<u32>() {
                            Ok(value) => {
                                if value < 60 {
                                    minute = value;
                                    index += 2;
                                    status = DateFormatStatus::Default;
                                }
                            }
                            Err(e) => {
                                return Err(DataFusionError::Execution(format!(
                                    "Unable to parse minute: {}",
                                    e
                                )))
                            }
                        },
                        'S' | 's' => match date_str[index..index + 2].parse::<u32>() {
                            Ok(value) => {
                                if value < 60 {
                                    second = value;
                                    index += 2;
                                    status = DateFormatStatus::Default;
                                }
                            }
                            Err(e) => {
                                return Err(DataFusionError::Execution(format!(
                                    "Unable to parse seconds: {}",
                                    e
                                )))
                            }
                        },
                        'x' => match date_str[index..index + 4].parse::<i32>() {
                            Ok(value) => {
                                year = value;
                                index += 4;
                                status = DateFormatStatus::Default;
                            }
                            Err(e) => {
                                return Err(DataFusionError::Execution(format!(
                                    "Unable to parse year: {}",
                                    e
                                )));
                            }
                        },
                        'v' => match date_str[index..index + 2].parse::<u32>() {
                            Ok(value) => {
                                if value >= 1 && value <= 53 {
                                    let iso_first_day =
                                        NaiveDate::from_isoywd_opt(year, 1, Weekday::Mon)
                                            .unwrap();
                                    let days_to_add = (value - 1) * 7;
                                    let final_date = iso_first_day
                                        + chrono::Duration::days(days_to_add as i64);

                                    year = final_date.year();
                                    month = final_date.month();
                                    day = final_date.day();

                                    index += 2;
                                    status = DateFormatStatus::Default;
                                } else {
                                    return Err(DataFusionError::Execution(
                                        "Invalid week number".into(),
                                    ));
                                }
                            }
                            Err(e) => {
                                return Err(DataFusionError::Execution(format!(
                                    "Unable to parse week number: {}",
                                    e
                                )));
                            }
                        },
                        'f' => {
                            let mut fraction_str = String::new();
                            let mut length = 0;

                            while index + length < date_str.len()
                                && date_str.as_bytes()[index + length].is_ascii_digit()
                            {
                                fraction_str
                                    .push(date_str.as_bytes()[index + length] as char);
                                length += 1;

                                if length > 9 {
                                    return Err(DataFusionError::Execution(
                                        "Fractional second precision not supported"
                                            .into(),
                                    ));
                                }
                            }

                            match fraction_str.parse::<String>() {
                                Ok(value) => {
                                    if length >= 3 {
                                        millis = value[..3].parse().unwrap_or(0);
                                        index += length;
                                        status = DateFormatStatus::Default;
                                    } else {
                                        match length {
                                            1 => {
                                                millis =
                                                    value.parse::<u32>().unwrap() * 100
                                            }
                                            2 => {
                                                millis =
                                                    value.parse::<u32>().unwrap() * 10
                                            }
                                            _ => (),
                                        }
                                    };
                                }
                                Err(e) => {
                                    return Err(DataFusionError::Execution(format!(
                                        "Unable to parse millis: {}",
                                        e
                                    )))
                                }
                            }
                        }
                        'M' | 'b' => {
                            let mut end_index = index;
                            while end_index < date_str.len()
                                && date_str.as_bytes()[end_index].is_ascii_alphabetic()
                            {
                                end_index += 1;
                            }

                            let month_num = &date_str[index..end_index];

                            match month_map.get(month_num.to_lowercase().as_str()) {
                                Some(&num) => {
                                    month = num;
                                    index = end_index;
                                    status = DateFormatStatus::Default;
                                }
                                None => {
                                    return Err(DataFusionError::Execution(format!(
                                        "Invalid full month name: {}",
                                        month_num
                                    )));
                                }
                            }
                        }
                        'W' | 'a' => {
                            let mut end_index = index;
                            while end_index < date_str.len()
                                && date_str.as_bytes()[end_index].is_ascii_alphabetic()
                            {
                                end_index += 1;
                            }
                            let full_weekday = &date_str[index..end_index];
                            match weekday_map.get(full_weekday.to_lowercase().as_str()) {
                                Some(&weekday) => {
                                    let parsed_date =
                                        NaiveDate::from_ymd_opt(year, month, day)
                                            .unwrap();
                                    let adjusted_date =
                                        adjust_to_weekday(parsed_date, weekday);
                                    year = adjusted_date.year();
                                    month = adjusted_date.month();
                                    day = adjusted_date.day();
                                    index = end_index;
                                    status = DateFormatStatus::Default;
                                }
                                None => {
                                    return Err(DataFusionError::Execution(format!(
                                        "Invalid full weekday name: {}",
                                        full_weekday
                                    )))
                                }
                            }
                        }
                        'j' => match parse_next_number(&date_str, index) {
                            Ok((value, end_index)) => {
                                if value >= 1 && value <= 366 {
                                    let first_day_of_year =
                                        NaiveDate::from_ymd_opt(year, 1, 1).unwrap();
                                    let date = first_day_of_year
                                        .checked_add_signed(chrono::Duration::days(
                                            value as i64 - 1,
                                        ))
                                        .ok_or_else(|| {
                                            DataFusionError::Execution(
                                                "Invalid day of year".into(),
                                            )
                                        })?;

                                    year = date.year();
                                    month = date.month();
                                    day = date.day();
                                    index = end_index;
                                    status = DateFormatStatus::Default;
                                } else {
                                    return Err(DataFusionError::Execution(
                                        "Invalid day of year".into(),
                                    ));
                                }
                            }
                            Err(e) => {
                                return Err(DataFusionError::Execution(format!(
                                    "Unable to parse day of year: {}",
                                    e
                                )))
                            }
                        },
                        'r' => {
                            let mut end_index = index;
                            while end_index < date_str.len()
                                && date_str.as_bytes()[end_index] != b' '
                            {
                                end_index += 1;
                            }

                            end_index += 1;

                            while end_index < date_str.len()
                                && date_str.as_bytes()[end_index].is_ascii_alphabetic()
                            {
                                end_index += 1;
                            }
                            let time_str = &date_str[index..end_index];
                            let time_parts: Vec<&str> =
                                time_str.split_whitespace().collect();

                            if time_parts.len() == 2 {
                                let time_components: Vec<&str> =
                                    time_parts[0].split(':').collect();
                                if time_components.len() == 3 {
                                    match (
                                        time_components[0].parse::<u32>(),
                                        time_components[1].parse::<u32>(),
                                        time_components[2].parse::<u32>(),
                                        time_parts[1].to_uppercase().as_str(),
                                    ) {
                                        (Ok(h), Ok(m), Ok(s), "AM" | "PM") => {
                                            hour = if h == 12 && time_parts[1] == "AM" {
                                                0
                                            } else if h != 12 && time_parts[1] == "PM" {
                                                h + 12
                                            } else {
                                                h
                                            };
                                            minute = m;
                                            second = s;
                                            index = end_index;
                                            status = DateFormatStatus::Default;
                                        }
                                        _ => {
                                            return Err(DataFusionError::Execution(
                                                "Invalid time format for %r".into(),
                                            ))
                                        }
                                    }
                                } else {
                                    return Err(DataFusionError::Execution(
                                        "Invalid time format for %r".into(),
                                    ));
                                }
                            } else {
                                return Err(DataFusionError::Execution(
                                    "Invalid time format for %r".into(),
                                ));
                            }
                        }
                        'T' => {
                            let mut end_index = index;
                            while end_index < date_str.len()
                                && date_str.as_bytes()[end_index] != b' '
                                && date_str.as_bytes()[end_index] != b'%'
                            {
                                end_index += 1;
                            }
                            let time_str = &date_str[index..end_index];
                            let time_components: Vec<&str> =
                                time_str.split(':').collect();

                            if time_components.len() == 3 {
                                match (
                                    time_components[0].parse::<u32>(),
                                    time_components[1].parse::<u32>(),
                                    time_components[2].parse::<u32>(),
                                ) {
                                    (Ok(h), Ok(m), Ok(s)) => {
                                        if h < 24 && m < 60 && s < 60 {
                                            hour = h;
                                            minute = m;
                                            second = s;
                                            index = end_index;
                                            status = DateFormatStatus::Default;
                                        } else {
                                            return Err(DataFusionError::Execution(
                                                "Invalid time format for %T".into(),
                                            ));
                                        }
                                    }
                                    _ => {
                                        return Err(DataFusionError::Execution(
                                            "Invalid time format for %T".into(),
                                        ))
                                    }
                                }
                            } else {
                                return Err(DataFusionError::Execution(
                                    "Invalid time format for %T".into(),
                                ));
                            }
                        }
                        'D' | 'U' | 'u' | 'V' | 'w' | 'X' => {
                            return Err(DataFusionError::Execution(format!(
                                "Specifier '{}' not supported",
                                c
                            )));
                        }
                        _ => {
                            return Err(DataFusionError::Execution(format!(
                                "invalid specifier'{}'",
                                c
                            )));
                        }
                    },
                }
            }
        }

        let date = NaiveDate::from_ymd_opt(year, month, day).ok_or_else(|| {
            ArrowError::ParseError(format!("Invalid date: {}-{}-{}", year, month, day))
        })?;
        let time = NaiveTime::from_hms_milli_opt(hour, minute, second, millis)
            .ok_or_else(|| {
                ArrowError::ParseError(format!(
                    "Invalid time: {}:{}:{}.{}",
                    hour, minute, second, millis
                ))
            })?;
        let datetime = NaiveDateTime::new(date, time);

        timestamps.push(datetime.timestamp_millis());

        Ok(Arc::new(TimestampMillisecondArray::from(timestamps)) as ArrayRef)
    }
}

fn adjust_to_weekday(current_date: NaiveDate, target_weekday: Weekday) -> NaiveDate {
    let current_weekday = current_date.weekday();
    let days_until_target = (target_weekday.num_days_from_sunday() as i64
        - current_weekday.num_days_from_sunday() as i64
        + 7)
        % 7;
    current_date + Duration::days(days_until_target)
}

fn parse_next_number(
    date_str: &str,
    start_index: usize,
) -> Result<(u32, usize), DataFusionError> {
    let mut end_index = start_index;
    while end_index < date_str.len() && date_str.as_bytes()[end_index].is_ascii_digit() {
        end_index += 1;
    }
    let number_str = &date_str[start_index..end_index];
    match number_str.parse::<u32>() {
        Ok(number) => Ok((number, end_index)),
        Err(e) => Err(DataFusionError::Execution(format!(
            "Unable to parse number: {}",
            e
        ))),
    }
}

#[derive(Debug)]
pub struct DateFormatFunction;

impl ScalarFunctionDef for DateFormatFunction {
    fn name(&self) -> &str {
        "date_format"
    }

    fn signature(&self) -> Signature {
        Signature::exact(
            vec![
                DataType::Timestamp(TimeUnit::Nanosecond, None),
                DataType::Utf8,
            ],
            Volatility::Immutable,
        )
    }

    fn return_type(&self) -> ReturnTypeFunction {
        Arc::new(move |_| Ok(Arc::new(DataType::Utf8)))
    }

    fn execute(&self, args: &[ArrayRef]) -> Result<ArrayRef> {
        // Validate input arguments
        assert_eq!(args.len(), 2, "date_format requires exactly two arguments");

        let timestamp_array = args[0]
            .as_any()
            .downcast_ref::<TimestampNanosecondArray>()
            .expect("Expected Timestamp nanosecond array for argument 1");
        let timestamp_milli_arr = timestamp_array
            .iter()
            .map(|timestamp| timestamp.map(|timestamp| timestamp / 1_000_000))
            .collect::<TimestampMillisecondArray>();
        let format_array = args[1]
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("Expected String array for argument 2");

        let mut formatted_dates = Vec::new();

        for i in 0..timestamp_milli_arr.len() {
            if timestamp_milli_arr.is_null(i) || format_array.is_null(i) {
                formatted_dates.push(None);
                continue;
            }

            let timestamp_millis = timestamp_milli_arr.value(i);
            let format_string = format_array.value(i);
            let naive_datetime = NaiveDateTime::from_timestamp_millis(timestamp_millis)
                .expect("Failed to convert timestamp to NaiveDateTime");
            let datetime: DateTime<Utc> =
                DateTime::from_naive_utc_and_offset(naive_datetime, Utc);

            let mut formatted_date = String::new();
            let mut chars = format_string.chars().peekable();

            while let Some(c) = chars.next() {
                if c == '%' {
                    if let Some(format_char) = chars.next() {
                        formatted_date += &match format_char {
                            'a' => datetime.format("%a").to_string(),
                            'b' => datetime.format("%b").to_string(),
                            'c' => datetime.month().to_string(),
                            'd' => datetime.format("%d").to_string(),
                            'e' => datetime.format("%e").to_string(),
                            'f' => {
                                let milliseconds = datetime.timestamp_subsec_millis();
                                format!("{:03}000", milliseconds)
                            }
                            'H' => datetime.format("%H").to_string(),
                            'h' => {
                                let hour = datetime.hour12().1;
                                format!("{:02}", hour)
                            }
                            'I' => datetime.format("%I").to_string(),
                            'i' => {
                                let minutes = datetime.minute();
                                format!("{:02}", minutes)
                            }
                            'j' => datetime.format("%j").to_string(),
                            'k' => datetime.format("%k").to_string(),
                            'l' => datetime.format("%l").to_string(),
                            'M' => datetime.format("%B").to_string(),
                            'm' => datetime.format("%m").to_string(),
                            'p' => datetime.format("%p").to_string(),
                            'r' => datetime.format("%r").to_string(),
                            'S' => datetime.format("%S").to_string(),
                            's' => datetime.format("%S").to_string(),
                            'T' => datetime.format("%T").to_string(),
                            'v' => datetime.format("%V").to_string(),
                            'W' => datetime.format("%A").to_string(),
                            'x' => datetime.format("%G").to_string(),
                            'Y' => datetime.format("%Y").to_string(),
                            'y' => datetime.format("%y").to_string(),
                            'D' | 'U' | 'u' | 'V' | 'w' | 'X' => {
                                return Err(DataFusionError::Execution(format!(
                                    "%{} is not currently supported in the date format string in Trino (Presto).",
                                    format_char
                                )));
                            }
                            _ => format_char.to_string(),
                        };
                    }
                } else {
                    formatted_date.push(c);
                }
            }
            formatted_dates.push(Some(formatted_date));
        }

        Ok(Arc::new(StringArray::from(formatted_dates)) as ArrayRef)
    }
}
#[derive(Debug)]
pub struct FormatDatetimeFunction;

impl ScalarFunctionDef for FormatDatetimeFunction {
    fn name(&self) -> &str {
        "format_datetime"
    }

    fn signature(&self) -> Signature {
        Signature::exact(
            vec![
                DataType::Timestamp(TimeUnit::Nanosecond, None),
                DataType::Utf8,
            ],
            Volatility::Immutable,
        )
    }

    fn return_type(&self) -> ReturnTypeFunction {
        Arc::new(move |_| Ok(Arc::new(DataType::Utf8)))
    }

    fn execute(&self, args: &[ArrayRef]) -> Result<ArrayRef> {
        // Validate input arguments
        assert_eq!(args.len(), 2, "date_format requires exactly two arguments");

        let timestamp_array = args[0]
            .as_any()
            .downcast_ref::<TimestampNanosecondArray>()
            .expect("Expected Timestamp nanosecond array for argument 1");
        let timestamp_milli_arr = timestamp_array
            .iter()
            .map(|timestamp| timestamp.map(|timestamp| timestamp / 1_000_000))
            .collect::<TimestampMillisecondArray>();
        let format_array = args[1]
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("Expected String array for argument 2");

        let mut formatted_dates = Vec::new();

        for i in 0..timestamp_milli_arr.len() {
            if timestamp_milli_arr.is_null(i) || format_array.is_null(i) {
                formatted_dates.push(None);
                continue;
            }

            let timestamp_millis = timestamp_milli_arr.value(i);
            let format_string = format_array.value(i).to_string();
            let naive_datetime = NaiveDateTime::from_timestamp_millis(timestamp_millis)
                .expect("Failed to convert timestamp to NaiveDateTime");
            let datetime: DateTime<Utc> =
                DateTime::from_naive_utc_and_offset(naive_datetime, Utc);

            let mut formatted_date = String::new();
            let mut chars = format_string.chars().peekable();
            let mut escape_mode = false;

            while let Some(c) = chars.next() {
                if c == '\'' && chars.peek() != Some(&'\'') {
                    escape_mode = !escape_mode;
                    continue;
                }

                if escape_mode {
                    formatted_date.push(c);
                    continue;
                }

                // Repeating character logic: count consecutive characters
                let mut count = 1;
                while chars.peek() == Some(&c) {
                    count += 1;
                    chars.next(); // consume the character
                }

                let formatted_component = match c {
                    'G' => {
                        if datetime.year() < 1 {
                            "BC".to_string()
                        } else {
                            "AD".to_string()
                        }
                    }
                    'C' => format!("{:0>width$}", datetime.year() / 100, width = count),
                    'Y' => {
                        if count == 2 {
                            format!("{:02}", datetime.year() % 100)
                        } else {
                            format!("{:0>width$}", datetime.format("%Y"), width = count)
                        }
                    }
                    'x' => format!("{:0>width$}", datetime.format("%G"), width = count),
                    'w' => format!("{:0>width$}", datetime.format("%V"), width = count),
                    'e' => format!("{:0>width$}", datetime.format("%u"), width = count),
                    'E' => format!("{:0>width$}", datetime.format("%a"), width = count),
                    'y' => format!("{:0>width$}", datetime.format("%Y"), width = count),
                    'D' => format!("{:0>width$}", datetime.format("%j"), width = count),
                    'M' => format!("{:0>width$}", datetime.format("%m"), width = count),
                    'd' => format!("{:0>width$}", datetime.format("%d"), width = count),
                    'a' => format!("{:0>width$}", datetime.format("%p"), width = count),
                    'K' => {
                        let hour_of_halfday_k = datetime.hour() % 12;
                        format!("{:0>width$}", hour_of_halfday_k, width = count)
                    }
                    'h' => format!("{:0>width$}", datetime.format("%I"), width = count),
                    'H' => format!("{:0>width$}", datetime.format("%H"), width = count),
                    'k' => {
                        let k24_midnight = if datetime.hour() == 0 {
                            24
                        } else {
                            datetime.hour()
                        }; // Midnight
                        format!("{:0>width$}", k24_midnight, width = count)
                    }
                    'm' => format!("{:0>width$}", datetime.format("%M"), width = count),
                    's' => format!("{:0>width$}", datetime.format("%S"), width = count),
                    'S' => {
                        let milliseconds = datetime.timestamp_subsec_millis();
                        match count {
                            1 => format!("{}", milliseconds / 100),
                            2 => format!("{}", milliseconds / 10),
                            _ => format!("{:0<width$}", milliseconds, width = count),
                        }
                    }
                    '\'' => "'".to_string(), // Handle literal text delimiter
                    _ => c.to_string().repeat(count),
                };

                formatted_date += &formatted_component;
            }
            formatted_dates.push(Some(formatted_date));
        }

        Ok(Arc::new(StringArray::from(formatted_dates)) as ArrayRef)
    }
}

// Function package declaration
pub struct FunctionPackage;

impl ScalarFunctionPackage for FunctionPackage {
    fn functions(&self) -> Vec<Box<dyn ScalarFunctionDef>> {
        vec![
            Box::new(HumanReadableSecondsFunction),
            Box::new(CurrentTimeFunction),
            Box::new(CurrentTimestampFunction),
            Box::new(CurrentTimestampPFunction),
            Box::new(CurrentTimezoneFunction),
            Box::new(LocaltimeFunction),
            Box::new(LocaltimestampFunction),
            Box::new(LocaltimestampPFunction),
            Box::new(ToMilliSecondsFunction),
            Box::new(ToIso8601Function),
            Box::new(FromIso8601DateFunction),
            Box::new(UnixTimeFunction),
            Box::new(FromUnixtimeFunction),
            Box::new(FromUnixtimeNanosFunction),
            Box::new(DayFunction),
            Box::new(DateTruncFunction),
            Box::new(DateDiffFunction),
            Box::new(ParseDurationFunction),
            Box::new(DateAddFunction),
            Box::new(LastDayOfMonthFunction),
            Box::new(DateParseFunction),
            Box::new(YearOfWeekFunction),
            Box::new(DayOfYearFunction),
            Box::new(DayOfWeekFunction),
            Box::new(HourFunction),
            Box::new(MillisecondFunction),
            Box::new(MinuteFunction),
            Box::new(SecondFunction),
            Box::new(WeekFunction),
            Box::new(YearFunction),
            Box::new(MonthFunction),
            Box::new(YearFunction),
            Box::new(QuarterFunction),
            Box::new(DateFormatFunction),
            Box::new(FormatDatetimeFunction),
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
    use chrono::{Local, Offset, Utc};
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

    fn roughly_equal_to_now(millisecond: i64) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;
        (millisecond - now).abs() <= 1_000_000
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
        let now_local = Local::now();
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
    async fn test_to_milliseconds() -> Result<()> {
        test_expression!("to_milliseconds(interval '1' day)", "86400000");
        test_expression!("to_milliseconds(interval '1' hour)", "3600000");
        test_expression!("to_milliseconds(interval '10' day)", "864000000");
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

    #[tokio::test]
    async fn date_trunc() -> Result<()> {
        // // Date
        test_expression!("date_trunc('day',DATE '2001-08-22')", "2001-08-22");
        test_expression!("date_trunc('week',DATE '2001-08-22')", "2001-08-20");
        test_expression!("date_trunc('quarter',DATE '2001-08-22')", "2001-07-01");
        test_expression!("date_trunc('month',DATE '2023-02-15')", "2023-02-01");
        test_expression!("date_trunc('year',DATE '2023-02-15')", "2023-01-01");
        // Timestamp
        test_expression!(
            "date_trunc( 'second',TIMESTAMP '2023-02-15T08:30:01')",
            "2023-02-15T08:30:01"
        );
        test_expression!(
            "date_trunc( 'minute',TIMESTAMP '2023-02-15T08:30:01')",
            "2023-02-15T08:30:00"
        );
        test_expression!(
            "date_trunc( 'hour',TIMESTAMP '2023-02-15T08:30:01')",
            "2023-02-15T08:00:00"
        );
        test_expression!(
            "date_trunc( 'day',TIMESTAMP '2023-02-15T08:30:01')",
            "2023-02-15T00:00:00"
        );
        test_expression!(
            "date_trunc( 'week',TIMESTAMP '2023-02-15T08:30:01')",
            "2023-02-13T00:00:00"
        );
        test_expression!(
            "date_trunc( 'month',TIMESTAMP '2023-02-15T08:30:01')",
            "2023-02-01T00:00:00"
        );
        test_expression!(
            "date_trunc( 'quarter',TIMESTAMP '2023-02-15T08:30:01')",
            "2023-01-01T00:00:00"
        );
        test_expression!(
            "date_trunc( 'year',TIMESTAMP '2023-02-15T08:30:01')",
            "2023-01-01T00:00:00"
        );
        // Time
        test_expression!("date_trunc('hour',TIME '08:09:10.123')", "08:00:00");
        test_expression!("date_trunc('minute',TIME '08:09:10.123')", "08:09:00");
        test_expression!("date_trunc('second',TIME '08:09:10.123')", "08:09:10");
        Ok(())
    }

    #[tokio::test]
    async fn test_date_diff() -> Result<()> {
        test_expression!("date_diff('second', TIMESTAMP '2020-03-01 00:00:00', TIMESTAMP '2020-03-02 00:00:00')","86400");
        test_expression!("date_diff('hour', TIMESTAMP '2020-03-01 00:00:00 UTC', TIMESTAMP '2020-03-02 00:00:00 UTC')","24");
        test_expression!("date_diff('second', TIMESTAMP '2020-06-01 12:30:45.000000000', TIMESTAMP '2020-06-02 12:30:45.123456789')","86400");
        test_expression!(
            "date_diff('day', DATE '2020-03-01', DATE '2020-03-02')",
            "1"
        );
        test_expression!("date_diff('millisecond', TIMESTAMP '2020-06-01 12:30:45.000000000', TIMESTAMP '2020-06-02 12:30:45.123456789')","86400123");
        Ok(())
    }

    #[tokio::test]
    async fn test_parse_duration() -> Result<()> {
        test_expression!("parse_duration('42.8ms')", "0 00:00:00.043");
        test_expression!("parse_duration('42.8ns')", "0 00:00:00.000");
        test_expression!("parse_duration('3.81 d')", "3 19:26:24.000");
        test_expression!("parse_duration('5m')", "0 00:05:00.000");
        Ok(())
    }
    #[tokio::test]
    async fn test_date_add() -> Result<()> {
        // date
        test_expression!("date_add('day', -1, DATE '2020-03-01')", "2020-02-29");
        test_expression!("date_add('week', 2, DATE '2020-03-01')", "2020-03-15");
        test_expression!("date_add('month', 3, DATE '2020-03-01')", "2020-06-01");
        test_expression!("date_add('quarter', -2, DATE '2020-03-01')", "2019-09-01");
        test_expression!("date_add('year', 4, DATE '2020-03-01')", "2024-03-01");
        // time
        test_expression!(
            "date_add('millisecond', 86, TIME '00:00:00.000')",
            "00:00:00.086"
        );
        test_expression!(
            "date_add('second', 86, TIME '00:00:00.006')",
            "00:01:26.006"
        );
        test_expression!(
            "date_add('minute', -9, TIME '00:09:00.006')",
            "00:00:00.006"
        );
        test_expression!("date_add('hour', 9, TIME '00:00:00.006')", "09:00:00.006");
        // timestamp
        test_expression!(
            "date_add('millisecond', 100000, TIMESTAMP '2020-03-01T00:00:00.000')",
            "2020-03-01T00:01:40"
        );
        test_expression!(
            "date_add('second', -86, TIMESTAMP '2020-03-01T00:01:27.003')",
            "2020-03-01T00:00:01.003"
        );
        test_expression!(
            "date_add('month', -5, TIMESTAMP '2020-03-01T00:00:00.000')",
            "2019-10-01T00:00:00"
        );
        test_expression!(
            "date_add('quarter', -2, TIMESTAMP '2020-03-01T00:00:00.100')",
            "2019-09-01T00:00:00.100"
        );
        Ok(())
    }
    #[tokio::test]
    async fn test_last_day_of_month() -> Result<()> {
        // Date
        test_expression!("last_day_of_month(DATE '2023-02-15')", "2023-02-28");
        // Timestamp
        test_expression!(
            "last_day_of_month( TIMESTAMP '2023-02-15T08:30:00')",
            "2023-02-28"
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_date_parse() -> Result<()> {
        test_expression!("date_parse('2013', '%Y')", "2013-01-01T00:00:00");
        test_expression!("date_parse('2013-05', '%Y-%m')", "2013-05-01T00:00:00");
        test_expression!("date_parse('2013-5', '%Y-%c')", "2013-05-01T00:00:00");
        test_expression!(
            "date_parse('2013-05-07', '%Y-%m-%d')",
            "2013-05-07T00:00:00"
        );
        test_expression!("date_parse('2013-05-1', '%Y-%m-%e')", "2013-05-01T00:00:00");
        test_expression!(
            "date_parse('2013-05-17 12:35:10', '%Y-%m-%d %h:%i:%s')",
            "2013-05-17T00:35:10"
        );
        test_expression!(
            "date_parse('2013-05-17 12:35:10 PM', '%Y-%m-%d %h:%i:%s %p')",
            "2013-05-17T12:35:10"
        );
        test_expression!(
            "date_parse('2013-05-17 12:35:10 AM', '%Y-%m-%d %h:%i:%s %p')",
            "2013-05-17T00:35:10"
        );
        test_expression!(
            "date_parse('2013-05-17 00:35:10', '%Y-%m-%d %H:%i:%s')",
            "2013-05-17T00:35:10"
        );
        test_expression!(
            "date_parse('2013-05-17 23:35:10', '%Y-%m-%d %H:%i:%s')",
            "2013-05-17T23:35:10"
        );
        test_expression!(
            "date_parse('abc 2013-05-17 fff 23:35:10 xyz','abc %Y-%m-%d fff %H:%i:%s xyz')",
            "2013-05-17T23:35:10"
        );
        test_expression!("date_parse('2023 24','%Y %y')", "2024-01-01T00:00:00");
        test_expression!("date_parse('2023 52','%x %v')", "2023-12-25T00:00:00");
        test_expression!("date_parse('2024 02','%x %v')", "2024-01-08T00:00:00");
        test_expression!("date_parse('01.1','%s.%f')", "1970-01-01T00:00:01.100");
        test_expression!("date_parse('01.01','%s.%f')", "1970-01-01T00:00:01.010");
        test_expression!("date_parse('01.2006','%s.%f')", "1970-01-01T00:00:01.200");
        test_expression!(
            "date_parse('59.123456789','%s.%f')",
            "1970-01-01T00:00:59.123"
        );
        test_expression!("date_parse('0','%k')", "1970-01-01T00:00:00");
        test_expression!(
            "date_parse('28-JAN-16 11.45.46.421000 PM','%d-%b-%y %l.%i.%s.%f %p')",
            "2016-01-28T23:45:46.421"
        );
        test_expression!(
            "date_parse('11-DEC-70 11.12.13.456000 AM','%d-%b-%y %l.%i.%s.%f %p')",
            "1970-12-11T11:12:13.456"
        );
        test_expression!(
            "date_parse('31-MAY-69 04.59.59.999000 AM','%d-%b-%y %l.%i.%s.%f %p')",
            "2069-05-31T04:59:59.999"
        );
        test_expression!(
            "date_parse('28-February-16 11.45.46.421000 PM','%d-%M-%y %l.%i.%s.%f %p')",
            "2016-02-28T23:45:46.421"
        );
        test_expression!("date_parse('1970-01-01','')", "1970-01-01T00:00:00");
        test_expression!(
            "date_parse('2024-01/Sunday','%Y-%m/%W')",
            "2024-01-07T00:00:00"
        );
        test_expression!(
            "date_parse('2024-01-13/Sunday','%Y-%m-%d/%W')",
            "2024-01-14T00:00:00"
        );
        test_expression!(
            "date_parse('2024-01-17/Saturday','%Y-%m-%d/%W')",
            "2024-01-20T00:00:00"
        );
        test_expression!(
            "date_parse('2024-01-17/Sunday','%Y-%m-%d/%W')",
            "2024-01-21T00:00:00"
        );
        test_expression!(
            "date_parse('2024-01-17/Sat','%Y-%m-%d/%a')",
            "2024-01-20T00:00:00"
        );
        test_expression!("date_parse('2024-61', '%Y-%j')", "2024-03-01T00:00:00");
        test_expression!("date_parse('12:20:34 AM', '%r')", "1970-01-01T00:20:34");
        test_expression!("date_parse('12:20:34 PM', '%r')", "1970-01-01T12:20:34");
        test_expression!(
            "date_parse('2024 10:20:34 AM', '%Y %r')",
            "2024-01-01T10:20:34"
        );
        test_expression!(
            "date_parse('2024 10:20:34', '%Y %T')",
            "2024-01-01T10:20:34"
        );
        Ok(())
    }
    #[tokio::test]
    async fn test_day_of_year() -> Result<()> {
        test_expression!("day_of_year(Date '2023-03-15')", "74");
        test_expression!("day_of_year(timestamp '2020-06-10 15:55:23.383345')", "162");
        Ok(())
    }
    #[tokio::test]
    async fn test_day_of_week() -> Result<()> {
        test_expression!("day_of_week(Date '2023-03-15')", "3");
        test_expression!("day_of_week(timestamp '2020-06-10 15:55:23.383345')", "3");
        Ok(())
    }
    #[tokio::test]
    async fn test_hour() -> Result<()> {
        test_expression!("hour(Date '2023-03-15')", "0"); // Date type does not have hour info, defaulting to 0
        test_expression!("hour(timestamp '2020-06-10 15:55:23.383345')", "15");
        Ok(())
    }

    #[tokio::test]
    async fn test_millisecond() -> Result<()> {
        test_expression!("millisecond(Date '2023-03-15')", "0"); // Date type does not have millisecond info, defaulting to 0
        test_expression!("millisecond(timestamp '2020-06-10 15:55:23.383345')", "383");
        Ok(())
    }
    #[tokio::test]
    async fn test_minute() -> Result<()> {
        test_expression!("minute(Date '2023-03-15')", "0"); // Date type does not have minute info, defaulting to 0
        test_expression!("minute(timestamp '2020-06-10 15:55:23.383345')", "55");
        Ok(())
    }
    #[tokio::test]
    async fn test_month() -> Result<()> {
        test_expression!("month(Date '2023-03-15')", "3");
        test_expression!("month(timestamp '2020-06-10 15:55:23.383345')", "6");
        Ok(())
    }

    #[tokio::test]
    async fn test_quarter() -> Result<()> {
        test_expression!("quarter(Date '2023-03-15')", "1");
        test_expression!("quarter(timestamp '2020-06-10 15:55:23.383345')", "2");
        Ok(())
    }
    #[tokio::test]
    async fn test_second() -> Result<()> {
        test_expression!("second(Date '2023-03-15')", "0"); // Date type does not have second info, defaulting to 0
        test_expression!("second(timestamp '2020-06-10 15:55:23.383345')", "23");
        Ok(())
    }
    #[tokio::test]
    async fn test_week() -> Result<()> {
        test_expression!("week(Date '2023-01-04')", "1");
        test_expression!("week(timestamp '2020-12-31 23:59:59.999999')", "53");
        Ok(())
    }
    #[tokio::test]
    async fn test_year() -> Result<()> {
        test_expression!("year(Date '2023-03-15')", "2023");
        test_expression!("year(timestamp '2020-06-10 15:55:23.383345')", "2020");
        Ok(())
    }
    #[tokio::test]
    async fn test_year_of_week() -> Result<()> {
        test_expression!("year_of_week(Date '2023-01-04')", "2023");
        test_expression!("year_of_week(timestamp '2020-12-31 23:59:59')", "2020");
        Ok(())
    }
    #[tokio::test]
    async fn test_date_format() -> Result<()> {
        test_expression!(
            "date_format(Timestamp '2022-10-20 05:10:00', '%m-%d-%Y %H')",
            "10-20-2022 05"
        );

        // timestamp
        test_expression!(
            "date_format(TIMESTAMP '2001-01-09 13:04:05.321', '%a')",
            "Tue"
        );
        test_expression!(
            "date_format(TIMESTAMP '2001-01-09 13:04:05.321', '%b')",
            "Jan"
        );
        test_expression!(
            "date_format(TIMESTAMP '2001-01-09 13:04:05.321', '%c')",
            "1"
        );
        test_expression!(
            "date_format(TIMESTAMP '2001-01-09 13:04:05.321', '%d')",
            "09"
        );
        test_expression!(
            "date_format(TIMESTAMP '2001-01-09 13:04:05.321', '%e')",
            " 9"
        );
        test_expression!(
            "date_format(TIMESTAMP '2001-01-09 13:04:05.321', '%f')",
            "321000"
        );
        test_expression!(
            "date_format(TIMESTAMP '2001-01-09 13:04:05.321', '%H')",
            "13"
        );
        test_expression!(
            "date_format(TIMESTAMP '2001-01-09 13:04:05.321', '%h')",
            "01"
        );
        test_expression!(
            "date_format(TIMESTAMP '2001-01-09 13:04:05.321', '%I')",
            "01"
        );
        test_expression!(
            "date_format(TIMESTAMP '2001-01-09 13:04:05.321', '%i')",
            "04"
        );
        test_expression!(
            "date_format(TIMESTAMP '2001-01-09 13:04:05.321', '%j')",
            "009"
        );
        test_expression!(
            "date_format(TIMESTAMP '2001-01-09 13:04:05.321', '%k')",
            "13"
        );
        test_expression!(
            "date_format(TIMESTAMP '2001-01-09 13:04:05.321', '%l')",
            " 1"
        );
        test_expression!(
            "date_format(TIMESTAMP '2001-01-09 13:04:05.321', '%M')",
            "January"
        );
        test_expression!(
            "date_format(TIMESTAMP '2001-01-09 13:04:05.321', '%m')",
            "01"
        );
        test_expression!(
            "date_format(TIMESTAMP '2001-01-09 13:04:05.321', '%p')",
            "PM"
        );
        test_expression!(
            "date_format(TIMESTAMP '2001-01-09 13:04:05.321', '%r')",
            "01:04:05 PM"
        );
        test_expression!(
            "date_format(TIMESTAMP '2001-01-09 13:04:05.321', '%S')",
            "05"
        );
        test_expression!(
            "date_format(TIMESTAMP '2001-01-09 13:04:05.321', '%s')",
            "05"
        );
        test_expression!(
            "date_format(TIMESTAMP '2001-01-09 13:04:05.321', '%T')",
            "13:04:05"
        );
        test_expression!(
            "date_format(TIMESTAMP '2001-01-09 13:04:05.321', '%v')",
            "02"
        );
        test_expression!(
            "date_format(TIMESTAMP '2001-01-09 13:04:05.321', '%W')",
            "Tuesday"
        );
        test_expression!(
            "date_format(TIMESTAMP '2018-12-31 13:04:05.321', '%x')",
            "2019"
        );
        test_expression!(
            "date_format(TIMESTAMP '2001-01-09 13:04:05.321', '%Y')",
            "2001"
        );
        test_expression!(
            "date_format(TIMESTAMP '2001-01-09 13:04:05.321', '%y')",
            "01"
        );
        test_expression!(
            "date_format(TIMESTAMP '2001-01-09 13:04:05.321', '%%')",
            "%"
        );
        test_expression!(
            "date_format(TIMESTAMP '2001-01-09 13:04:05.321', 'foo')",
            "foo"
        );
        test_expression!(
            "date_format(TIMESTAMP '2001-01-09 13:04:05.321', '%g')",
            "g"
        );
        test_expression!(
            "date_format(TIMESTAMP '2001-01-09 13:04:05.321', '%4')",
            "4"
        );
        test_expression!(
            "date_format(TIMESTAMP '2018-12-31 13:04:05.321', '%v %x %f')",
            "01 2019 321000"
        );
        test_expression!(
            "date_format(TIMESTAMP '2001-01-09 13:04:05.321', '%Yyear%mmonth%dday')",
            "2001year01month09day"
        );
        Ok(())
    }
    #[tokio::test]
    async fn test_format_datetime() -> Result<()> {
        // timestamp
        // Arrow DataFusion does not currently support dates before the Common Era.
        test_expression!(
            "format_datetime(TIMESTAMP '2001-12-31 13:04:05.321', 'G')",
            "AD"
        );
        test_expression!(
            "format_datetime(TIMESTAMP '2001-01-09 13:04:05.321', 'C')",
            "20"
        );
        test_expression!(
            "format_datetime(TIMESTAMP '2018-12-31 13:04:05.321', 'Y')",
            "2018"
        );
        test_expression!(
            "format_datetime(TIMESTAMP '2018-12-31 13:04:05.321', 'x')",
            "2019"
        );
        test_expression!(
            "format_datetime(TIMESTAMP '2001-01-09 13:04:05.321', 'w')",
            "02"
        );
        test_expression!(
            "format_datetime(TIMESTAMP '2001-01-09 13:04:05.321', 'e')",
            "2"
        );
        test_expression!(
            "format_datetime(TIMESTAMP '2001-01-09 13:04:05.321', 'E')",
            "Tue"
        );
        test_expression!(
            "format_datetime(TIMESTAMP '2001-01-09 13:04:05.321', 'y')",
            "2001"
        );
        test_expression!(
            "format_datetime(TIMESTAMP '2001-02-09 13:04:05.321', 'D')",
            "040"
        );
        test_expression!(
            "format_datetime(TIMESTAMP '2001-01-09 13:04:05.321', 'M')",
            "01"
        );
        test_expression!(
            "format_datetime(TIMESTAMP '2001-01-09 13:04:05.321', 'd')",
            "09"
        );
        test_expression!(
            "format_datetime(TIMESTAMP '2001-01-09 13:04:05.321', 'a')",
            "PM"
        );
        test_expression!(
            "format_datetime(TIMESTAMP '2001-01-09 13:04:05.321', 'K')",
            "1"
        );
        test_expression!(
            "format_datetime(TIMESTAMP '2001-01-09 12:04:05.321', 'K')",
            "0"
        );
        test_expression!(
            "format_datetime(TIMESTAMP '2001-01-09 00:04:05.321', 'h')",
            "12"
        );
        test_expression!(
            "format_datetime(TIMESTAMP '2001-01-09 13:04:05.321', 'H')",
            "13"
        );
        test_expression!(
            "format_datetime(TIMESTAMP '2001-01-09 13:04:05.321', 'k')",
            "13"
        );
        test_expression!(
            "format_datetime(TIMESTAMP '2001-01-09 00:04:05.321', 'k')",
            "24"
        );
        test_expression!(
            "format_datetime(TIMESTAMP '2001-01-09 00:04:05.321', 'm')",
            "04"
        );
        test_expression!(
            "format_datetime(TIMESTAMP '2001-01-09 13:04:05.321', 's')",
            "05"
        );
        test_expression!(
            "format_datetime(TIMESTAMP '2001-01-09 13:04:05.321', 'S')",
            "3"
        );
        test_expression!(
            "format_datetime(TIMESTAMP '2001-01-09 13:04:05.30', 'SS')",
            "30"
        );
        test_expression!(
            "format_datetime(TIMESTAMP '2001-01-09 13:04:05.321', 'SSSSSS')",
            "321000"
        );

        test_expression!(
            "format_datetime(TIMESTAMP '2001-01-09 13:04:05.321', 'AAA')",
            "AAA"
        );
        test_expression!(
            "format_datetime(TIMESTAMP '2001-01-09 13:04:05.321', '''')",
            ""
        );
        test_expression!(
            "format_datetime(TIMESTAMP '2023-01-09 13:04:05.321', '''''')",
            "'"
        );
        test_expression!(
            "format_datetime(TIMESTAMP '2023-01-09 13:04:05.321', '''Y''')",
            "Y"
        );
        test_expression!(
            "format_datetime(TIMESTAMP '2023-01-09 13:04:05.321', 'Y ''Year''')",
            "2023 Year"
        );

        test_expression!(
            "format_datetime(TIMESTAMP '2001-01-09 13:04:05.321', 'YY')",
            "01"
        );
        test_expression!(
            "format_datetime(TIMESTAMP '2023-01-09 13:04:05.321', 'Y Y')",
            "2023 2023"
        );
        test_expression!(
            "format_datetime(TIMESTAMP '2023-01-09 13:04:05.321', 'YYYYY')",
            "02023"
        );
        test_expression!(
            "format_datetime(TIMESTAMP '2023-01-09 13:04:05.321', '''yyyy-MM-dd HH:mm:ss''')",
            "yyyy-MM-dd HH:mm:ss"
        );
        test_expression!(
            "format_datetime(TIMESTAMP '2023-01-09 13:04:05.321', 'yyyy-MM-dd HH:mm:ss')",
            "2023-01-09 13:04:05"
        );
        test_expression!(
            "format_datetime(TIMESTAMP '2023-01-09 13:04:05.321', 'YYYY/MM/dd HH:mm')",
            "2023/01/09 13:04"
        );
        Ok(())
    }
}
