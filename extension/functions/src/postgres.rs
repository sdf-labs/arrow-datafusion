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

use arrow::array::{ArrayRef, BooleanArray, Int64Array, StringArray};
use arrow::datatypes::DataType;
use datafusion::error::Result;
use datafusion::logical_expr::Volatility;
use datafusion_common::cast::{as_int64_array, as_string_array};
use datafusion_expr::{
    ReturnTypeFunction, ScalarFunctionDef, ScalarFunctionPackage, Signature,
    TypeSignature,
};
use std::sync::Arc;

use regex::{Captures, Regex};
use unidecode::unidecode;

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
                TypeSignature::Exact(vec![
                    DataType::Utf8,
                    DataType::Utf8,
                    DataType::Int64,
                ]),
                TypeSignature::Exact(vec![
                    DataType::Utf8,
                    DataType::Utf8,
                    DataType::Int64,
                    DataType::Utf8,
                ]),
            ],
            Volatility::Immutable,
        )
    }

    fn return_type(&self) -> ReturnTypeFunction {
        let return_type = Arc::new(DataType::Int64);
        Arc::new(move |_| Ok(return_type.clone()))
    }

    fn execute(&self, args: &[ArrayRef]) -> Result<ArrayRef> {
        assert!(args.len() >= 2);

        let input = as_string_array(&args[0]).expect("cast failed");
        let pattern = as_string_array(&args[1]).expect("cast failed");

        let start = if args.len() >= 3 {
            as_int64_array(&args[2])
                .expect("cast failed")
                .iter()
                .next()
                .unwrap_or(None)
        } else {
            None
        };

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

        let array = input
            .into_iter()
            .zip(pattern.into_iter())
            .map(|(text, pat)| {
                if let (Some(text), Some(pat)) = (text, pat) {
                    let text = &text[start.unwrap_or(0) as usize..];
                    let re = Regex::new(&format!("{}{}", flag_string, pat))
                        .expect("Invalid regex pattern");
                    Some(re.find_iter(text).count() as i64)
                } else {
                    None
                }
            })
            .collect::<Int64Array>();
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
                TypeSignature::Exact(vec![
                    DataType::Utf8,
                    DataType::Utf8,
                    DataType::Utf8,
                ]),
            ],
            Volatility::Immutable,
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

        let array = input
            .into_iter()
            .zip(pattern.into_iter())
            .map(|(text, pat)| {
                if let (Some(text), Some(pat)) = (text, pat) {
                    let re = Regex::new(&format!("{}{}", flag_string, pat))
                        .expect("Invalid regex pattern");
                    Some(re.is_match(text))
                } else {
                    None
                }
            })
            .collect::<BooleanArray>();

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
                TypeSignature::Exact(vec![
                    DataType::Utf8,
                    DataType::Utf8,
                    DataType::Utf8,
                ]),
                TypeSignature::Exact(vec![
                    DataType::Utf8,
                    DataType::Utf8,
                    DataType::Utf8,
                    DataType::Int64,
                    DataType::Int64,
                ]),
                TypeSignature::Exact(vec![
                    DataType::Utf8,
                    DataType::Utf8,
                    DataType::Utf8,
                    DataType::Int64,
                    DataType::Int64,
                    DataType::Utf8,
                ]),
            ],
            Volatility::Immutable,
        )
    }

    fn return_type(&self) -> ReturnTypeFunction {
        let return_type = Arc::new(DataType::Utf8);
        Arc::new(move |_| Ok(return_type.clone()))
    }

    fn execute(&self, args: &[ArrayRef]) -> Result<ArrayRef> {
        assert!(args.len() >= 3);

        let input = as_string_array(&args[0]).expect("cast failed");
        let pattern = as_string_array(&args[1]).expect("cast failed");
        let replacement = as_string_array(&args[2]).expect("cast failed");
        let start = if args.len() > 3 {
            as_int64_array(&args[3])
                .expect("cast failed")
                .iter()
                .next()
                .unwrap_or(Some(0))
        } else {
            Some(0)
        };

        let n = if args.len() > 4 {
            as_int64_array(&args[4])
                .expect("cast failed")
                .iter()
                .next()
                .unwrap_or(Some(0))
        } else {
            Some(0)
        };

        let flag_string = if args.len() > 5 {
            let flags_array = as_string_array(&args[5]).expect("cast failed");
            if let Some(flag) = flags_array.iter().next().unwrap_or(None) {
                format!("(?{})", flag)
            } else {
                String::new()
            }
        } else {
            String::new()
        };

        let array = input
            .into_iter()
            .zip(pattern.into_iter().zip(replacement.into_iter()))
            .map(|(text, (pat, rep))| {
                if let (Some(text), Some(pat), Some(rep)) = (text, pat, rep) {
                    let start_pos =
                        start.map_or(0, |s| if s > 0 { s as usize - 1 } else { 0 });
                    let prefix = &text[..start_pos];
                    let rest_text = &text[start_pos..];
                    let re = Regex::new(&format!("{}{}", flag_string, pat))
                        .expect("Invalid regex pattern");

                    if n == Some(0) {
                        Some(format!("{}{}", prefix, re.replace_all(rest_text, rep)))
                    } else {
                        let mut replaced_text = String::from(prefix);
                        let mut match_count = 0;
                        let mut last_end = 0;
                        for cap in re.captures_iter(rest_text) {
                            match_count += 1;
                            replaced_text.push_str(
                                &rest_text[last_end..cap.get(0).unwrap().start()],
                            );
                            if match_count == n.unwrap() {
                                replaced_text.push_str(rep);
                                last_end = cap.get(0).unwrap().end();
                                break;
                            } else {
                                replaced_text.push_str(&cap.get(0).unwrap().as_str());
                                last_end = cap.get(0).unwrap().end();
                            }
                        }
                        replaced_text.push_str(&rest_text[last_end..]);
                        Some(replaced_text)
                    }
                } else {
                    None
                }
            })
            .collect::<StringArray>();

        Ok(Arc::new(array) as ArrayRef)
    }
}

#[derive(Debug)]
pub struct NormalizeFunction;

impl ScalarFunctionDef for NormalizeFunction {
    fn name(&self) -> &str {
        "normalize"
    }

    fn signature(&self) -> Signature {
        Signature::one_of(
            vec![
                TypeSignature::Exact(vec![DataType::Utf8]),
                TypeSignature::Exact(vec![DataType::Utf8, DataType::Utf8]),
            ],
            Volatility::Immutable,
        )
    }

    fn return_type(&self) -> ReturnTypeFunction {
        let return_type = Arc::new(DataType::Utf8);
        Arc::new(move |_| Ok(return_type.clone()))
    }

    fn execute(&self, args: &[ArrayRef]) -> Result<ArrayRef> {
        assert!(args.len() >= 1);

        let input = as_string_array(&args[0]).expect("cast failed");
        let form = if args.len() >= 2 {
            as_string_array(&args[1])
                .expect("cast failed")
                .iter()
                .next()
                .unwrap_or(None)
        } else {
            Some("NFC")
        };

        let array = input
            .into_iter()
            .map(|text| {
                if let Some(text) = text {
                    let normalized_text = match form.as_deref() {
                        Some("NFC") => {
                            unicode_normalization::UnicodeNormalization::nfc(text.chars())
                                .collect::<String>()
                        }
                        Some("NFD") => {
                            unicode_normalization::UnicodeNormalization::nfd(text.chars())
                                .collect::<String>()
                        }
                        Some("NFKC") => {
                            unicode_normalization::UnicodeNormalization::nfkc(
                                text.chars(),
                            )
                            .collect::<String>()
                        }
                        Some("NFKD") => {
                            unicode_normalization::UnicodeNormalization::nfkd(
                                text.chars(),
                            )
                            .collect::<String>()
                        }
                        _ => text.to_string(),
                    };
                    Some(normalized_text)
                } else {
                    None
                }
            })
            .collect::<StringArray>();
        Ok(Arc::new(array) as ArrayRef)
    }
}

#[derive(Debug)]
pub struct ToAsciiFunction;

impl ScalarFunctionDef for ToAsciiFunction {
    fn name(&self) -> &str {
        "to_ascii"
    }

    fn signature(&self) -> Signature {
        Signature::one_of(
            vec![
                TypeSignature::Exact(vec![DataType::Utf8]),
                TypeSignature::Exact(vec![DataType::Utf8, DataType::Utf8]),
                TypeSignature::Exact(vec![DataType::Utf8, DataType::Int64]),
            ],
            Volatility::Immutable,
        )
    }

    fn return_type(&self) -> ReturnTypeFunction {
        let return_type = Arc::new(DataType::Utf8);
        Arc::new(move |_| Ok(return_type.clone()))
    }
    fn execute(&self, args: &[ArrayRef]) -> Result<ArrayRef> {
        assert!(args.len() >= 1);

        let input = as_string_array(&args[0]).expect("cast failed");

        let array = input
            .into_iter()
            .map(|text| {
                if let Some(text) = text {
                    Some(unidecode(&text))
                } else {
                    None
                }
            })
            .collect::<StringArray>();
        Ok(Arc::new(array) as ArrayRef)
    }
}

#[derive(Debug)]
pub struct UniStrFunction;

impl ScalarFunctionDef for UniStrFunction {
    fn name(&self) -> &str {
        "unistr"
    }

    fn signature(&self) -> Signature {
        Signature::exact(vec![DataType::Utf8], Volatility::Immutable)
    }

    fn return_type(&self) -> ReturnTypeFunction {
        Arc::new(|_| Ok(Arc::new(DataType::Utf8)))
    }

    fn execute(&self, args: &[ArrayRef]) -> Result<ArrayRef> {
        assert_eq!(args.len(), 1);

        let input = as_string_array(&args[0]).expect("cast failed");

        let re = Regex::new(
            r"(?x)
            \\?\+([0-9A-Fa-f]{6}) |
            \\([0-9A-Fa-f]{4})   |
            \\u([0-9A-Fa-f]{4})  |
            \\U([0-9A-Fa-f]{8})  |
            \\\\",
        )
        .expect("Failed to create regex");

        let array = input
            .into_iter()
            .map(|opt_text| {
                opt_text.map(|text| {
                    re.replace_all(&text, |caps: &Captures| {
                        if let Some(hex) =
                            caps.get(1).or(caps.get(2)).or(caps.get(3)).or(caps.get(4))
                        {
                            let codepoint =
                                u32::from_str_radix(hex.as_str(), 16).unwrap();
                            char::from_u32(codepoint).unwrap().to_string()
                        } else {
                            "\\".to_string()
                        }
                    })
                    .to_string()
                })
            })
            .collect::<StringArray>();

        Ok(Arc::new(array) as ArrayRef)
    }
}

#[derive(Debug)]
pub struct QuoteNullableFunction;

impl ScalarFunctionDef for QuoteNullableFunction {
    fn name(&self) -> &str {
        "quote_nullable"
    }

    fn signature(&self) -> Signature {
        Signature::one_of(
            vec![TypeSignature::Exact(vec![DataType::Utf8])],
            Volatility::Immutable,
        )
    }

    fn return_type(&self) -> ReturnTypeFunction {
        let return_type = Arc::new(DataType::Utf8);
        Arc::new(move |_| Ok(return_type.clone()))
    }

    fn execute(&self, args: &[ArrayRef]) -> Result<ArrayRef> {
        assert_eq!(args.len(), 1);

        let input = as_string_array(&args[0]).expect("cast failed");

        let array = input
            .into_iter()
            .map(|text| {
                text.map(|t| format!("'{}'", t.replace("'", "''").replace("\\", "\\\\")))
            })
            .collect::<StringArray>();

        Ok(Arc::new(array) as ArrayRef)
    }
}

#[derive(Debug)]
pub struct QuoteLiteralFunction;

impl ScalarFunctionDef for QuoteLiteralFunction {
    fn name(&self) -> &str {
        "quote_literal"
    }

    fn signature(&self) -> Signature {
        Signature::one_of(
            vec![TypeSignature::Exact(vec![DataType::Utf8])],
            Volatility::Immutable,
        )
    }

    fn return_type(&self) -> ReturnTypeFunction {
        let return_type = Arc::new(DataType::Utf8);
        Arc::new(move |_| Ok(return_type.clone()))
    }

    fn execute(&self, args: &[ArrayRef]) -> Result<ArrayRef> {
        assert_eq!(args.len(), 1);

        let input = as_string_array(&args[0]).expect("cast failed");

        let array = input
            .into_iter()
            .map(|text| {
                text.map(|t| format!("'{}'", t.replace("'", "''").replace("\\", "\\\\")))
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
            Box::new(RegexpCountFunction),
            Box::new(RegexpLikeFunction),
            Box::new(RegexpReplaceFunction),
            Box::new(NormalizeFunction),
            Box::new(ToAsciiFunction),
            Box::new(UniStrFunction),
            Box::new(QuoteLiteralFunction),
            Box::new(QuoteNullableFunction),
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
        test_expression!("regexp_replace('Thomas', '.[mN]a.', 'M')", "ThM");
        test_expression!("regexp_replace('Thomas', '.', 'X', 3, 2)", "ThoXas");
        test_expression!("regexp_replace('banana', 'a', 'X', 1, 0)", "bXnXnX");
        test_expression!("regexp_replace('banana', 'a', 'X', 1, 1)", "bXnana");
        test_expression!("regexp_replace('bAnana', 'a', 'X', 1, 1, 'i')", "bXnana");
        Ok(())
    }

    #[tokio::test]
    async fn test_normalize() -> Result<()> {
        test_expression!("normalize('a\u{0308}bc')", "\u{00E4}bc");
        test_expression!("normalize('\u{00E4}bc', 'NFD')", "a\u{0308}bc");
        test_expression!("normalize('a\u{0308}bc', 'NFKC')", "\u{00E4}bc");
        test_expression!("normalize('\u{00E4}bc', 'NFKD')", "a\u{0308}bc");
        Ok(())
    }

    #[tokio::test]
    async fn test_to_ascii() -> Result<()> {
        test_expression!("to_ascii('Karél')", "Karel");
        test_expression!("to_ascii('álfa-βéta')", "alfa-beta");
        Ok(())
    }

    #[tokio::test]
    async fn test_unistr() -> Result<()> {
        test_expression!("unistr('d\\0061t\\+000061')", "data");
        test_expression!("unistr('d\\u0061t\\U00000061')", "data");
        test_expression!("unistr('Backslash \\\\')", "Backslash \\");
        Ok(())
    }

    #[tokio::test]
    async fn test_quote_nullable() -> Result<()> {
        test_expression!("quote_nullable('O''Reilly')", "'O''Reilly'");
        test_expression!("quote_nullable('O\\'Reilly')", "'O\\''Reilly'");
        test_expression!("quote_nullable('Hello World')", "'Hello World'");
        test_expression!("quote_nullable(null)", "null");
        Ok(())
    }

    #[tokio::test]
    async fn test_quote_literal() -> Result<()> {
        test_expression!("quote_literal('O''Reilly')", "'O''Reilly'");
        test_expression!("quote_literal('O\\'Reilly')", "'O\\''Reilly'");
        test_expression!("quote_literal('Hello World')", "'Hello World'");
        test_expression!("quote_literal(null)", "null");
        Ok(())
    }
}
