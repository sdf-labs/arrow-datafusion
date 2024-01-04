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
        ArrayRef, Int64Array, Time32MillisecondArray, TimestampMillisecondArray,
        TimestampNanosecondArray,
    },
    datatypes::{DataType, IntervalUnit, TimeUnit},
};
use chrono::{Datelike, Months, NaiveDateTime, NaiveTime, Timelike};
use datafusion::error::Result;
use datafusion_common::DataFusionError;
use datafusion_expr::{
    ReturnTypeFunction, ScalarFunctionDef, ScalarFunctionPackage, Signature,
    TypeSignature, Volatility,
};

use arrow::array::*;
use arrow::error::ArrowError;

use chrono::{Duration, NaiveDate, TimeZone, Utc};

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
        let current_time = chrono::Utc::now().time();
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
        Arc::new(move |_| Ok(Arc::new(DataType::Timestamp(TimeUnit::Microsecond, None))))
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
pub struct CurrentTimestampPFunction;

impl ScalarFunctionDef for CurrentTimestampPFunction {
    fn name(&self) -> &str {
        "current_timestamp_p"
    }

    fn signature(&self) -> Signature {
        Signature::exact(vec![DataType::Int64], Volatility::Immutable)
    }

    fn return_type(&self) -> ReturnTypeFunction {
        Arc::new(move |_| Ok(Arc::new(DataType::Timestamp(TimeUnit::Nanosecond, None))))
    }

    fn execute(&self, args: &[ArrayRef]) -> Result<ArrayRef> {
        let precision = args[0]
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0) as usize;

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|err| DataFusionError::Execution(err.to_string()))?;
        let nanos = now.as_nanos() as i64;

        let adjusted_nanos = if precision > 9 {
            nanos
        } else {
            let factor = 10_i64.pow(9 - precision as u32);
            (nanos / factor) * factor
        };

        let array = TimestampNanosecondArray::from(vec![Some(adjusted_nanos)]);
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
                                naive_date_time.weekday().num_days_from_sunday() as i64;
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
            Box::new(LastDayOfMonthFunction),
            Box::new(DateTruncFunction),
            Box::new(DateAddFunction),
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
    use chrono::Utc;
    use datafusion::error::Result;
    use datafusion::prelude::SessionContext;
    use datafusion_expr::ScalarFunctionDef;
    use tokio;

    use crate::{
        presto::{CurrentTimestampFunction, CurrentTimestampPFunction},
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

    #[tokio::test]
    async fn test_current_time() -> Result<()> {
        let current = Utc::now();
        let formatted = current.format("%H:%M:%S").to_string();
        test_expression!("current_time()", formatted);

        Ok(())
    }

    #[tokio::test]
    async fn test_to_milliseconds() -> Result<()> {
        test_expression!("to_milliseconds(interval '1' day)", "86400000");
        test_expression!("to_milliseconds(interval '1' hour)", "3600000");
        test_expression!("to_milliseconds(interval '10' day)", "864000000");
        Ok(())
    }

    fn roughly_equal_to_now(millisecond: i64) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;
        (millisecond - now).abs() <= 1
    }

    #[tokio::test]
    async fn test_current_timestamp() -> Result<()> {
        let current_timestamp_function = CurrentTimestampFunction {};
        let result = current_timestamp_function.execute(&[]).unwrap();
        let result_array = result
            .as_any()
            .downcast_ref::<TimestampMillisecondArray>()
            .unwrap()
            .value(0);
        assert!(roughly_equal_to_now(result_array));
        Ok(())
    }

    #[tokio::test]
    async fn test_current_timestamp_p() -> Result<()> {
        let function = CurrentTimestampPFunction {};
        let precision_array = Int64Array::from(vec![9]);
        let args = vec![Arc::new(precision_array) as ArrayRef];
        let result = function.execute(&args).unwrap();
        let result_array = result
            .as_any()
            .downcast_ref::<TimestampNanosecondArray>()
            .unwrap()
            .value(0);
        assert!(roughly_equal_to_now(result_array / 1_000_000));
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
        // Time
        test_expression!("date_trunc('hour',TIME '08:09:10.123')", "08:00:00");
        test_expression!("date_trunc('minute',TIME '08:09:10.123')", "08:09:00");
        test_expression!("date_trunc('second',TIME '08:09:10.123')", "08:09:10");
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
}
