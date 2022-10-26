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

//! SQL Parser
//!
//! Declares a SQL parser based on sqlparser that handles custom formats that we need.


use sqlparser::{
    ast::{ColumnDef, ColumnOptionDef, Statement as SQLStatement, TableConstraint, ObjectName, Ident},
    dialect::{keywords::Keyword, Dialect, GenericDialect},
    parser::{Parser, ParserError},
    tokenizer::{Token, Tokenizer},
};
use std::{
    collections::{HashSet, VecDeque},
    fmt, fs, path::{Path, PathBuf}, 
};

use lazy_static::lazy_static;
use std::sync::Mutex;
// use crate::{dialect::Dialect, parser::{Parser, ParserError}, ast::Statement, tokenizer::Token, keywords::Keyword};

lazy_static! {
    // absolute path. last component is the module name, 
    // collects all modules that have been visited so far
    pub static ref VISITED_FILES: Mutex<HashSet<String>> = Mutex::new(HashSet::new());
    // absolute paths, last one is the package name
    pub static ref VISITED_PACKAGES: Mutex<HashSet<String>> = Mutex::new(HashSet::new());
}

// Removes directory path and returns the file name; like path.filename, but for strings
pub fn  basename(path: &str) -> String{
    match path.rfind('/'){
        Some(i) => path[i+1..].to_owned(),
        None => path.to_owned()
    }
}
// Combines package and module path with suffix
pub fn  sql_filename(package_path: &str, module_path: &str) -> String{
    if package_path=="" {
        module_path.to_owned()+".sql"
    } else {
        package_path.to_owned()+"/"+module_path+".sql"
    }
}

static ROOT: &str = "sdf.pkg.yml";

pub fn find_package_file(starting_directory: &Path) -> Option<PathBuf> {
    let mut path: PathBuf = starting_directory.into();
    let root_filename = Path::new(ROOT);

    loop {
        path.push(root_filename);
        if path.is_file() {           
            break Some(path.canonicalize().unwrap());
        } 
        if !(path.pop() && path.pop()) { // remove file && remove parent
            break None;
        }
    }
}

pub fn find_package_path(starting_directory: &Path) -> Option<PathBuf>   {
    if let Some( path) = find_package_file(Path::new(&starting_directory)) {
        let mut tmp: PathBuf = path.into();
        tmp.pop();
        Some(tmp)
    } else {
       None
   }
}



// Use `Parser::expected` instead, if possible
macro_rules! parser_err {
    ($MSG:expr) => {
        Err(ParserError::ParserError($MSG.to_string()))
    };
}

fn parse_file_type(s: &str) -> Result<String, ParserError> {
    // let res = FILENAME.lock().unwrap().replace(String::from("other"));
    Ok(s.to_uppercase())
}

fn parse_file_compression_type(s: &str) -> Result<String, ParserError> {
    Ok(s.to_uppercase())
}

/// DataFusion extension DDL for `CREATE EXTERNAL TABLE`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CreateExternalTable {
    /// Table name
    pub name: String,
    /// Optional schema
    pub columns: Vec<ColumnDef>,
    /// File type (Parquet, NDJSON, CSV, etc)
    pub file_type: String,
    /// CSV Header row?
    pub has_header: bool,
    /// User defined delimiter for CSVs
    pub delimiter: char,
    /// Path to file
    pub location: String,
    /// Partition Columns
    pub table_partition_cols: Vec<String>,
    /// Option to not error if table already exists
    pub if_not_exists: bool,
    /// File compression type (GZIP, BZIP2)
    pub file_compression_type: String,
}

impl fmt::Display for CreateExternalTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CREATE EXTERNAL TABLE ")?;
        if self.if_not_exists {
            write!(f, "IF NOT EXSISTS ")?;
        }
        write!(f, "{} ", self.name)?;
        write!(f, "STORED AS {} ", self.file_type)?;
        write!(f, "LOCATION {} ", self.location)
    }
}

/// DataFusion extension DDL for `DESCRIBE TABLE`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DescribeTable {
    /// Table name
    pub table_name: String,
}

/// DataFusion Statement representations.
///
/// Tokens parsed by `DFParser` are converted into these values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Statement {
    /// ANSI SQL AST node with package_path module_path
    Statement(Box<SQLStatement>, String, String),
    /// Extension: `CREATE EXTERNAL TABLE` with package_path module_path
    CreateExternalTable(CreateExternalTable, String, String),
    /// Extension: `DESCRIBE TABLE` with package_path module_path
    DescribeTable(DescribeTable, String, String),

}

/// SQL Parser
#[allow(dead_code)]
pub struct DFParser<'a> {
    parser: Parser<'a>,
    package_path: String,
    module_path: String,
}

impl<'a> DFParser<'a> {
    /// Parse the specified tokens
    pub fn new(sql: &str) -> Result<Self, ParserError> {
        let dialect = &GenericDialect {};
        DFParser::new_with_dialect(sql, dialect)
    }

    /// Parse the specified tokens with dialect
    pub fn new_with_dialect(
        sql: &str,
        dialect: &'a dyn Dialect,
    ) -> Result<Self, ParserError> {
        let mut tokenizer = Tokenizer::new(dialect, sql);
        let tokens = tokenizer.tokenize()?;

        Ok(DFParser {
            parser: Parser::new(tokens, dialect),
            package_path : String::new(),
            module_path : String::new(),
        })
    }

    pub fn new_with_dialect_and_scope(
        sql: &str,
        dialect: &'a dyn Dialect,
        _filename: String,
        package_path: String,
        module_path: String,
    ) -> Result<Self, ParserError> {
        let mut tokenizer = Tokenizer::new(dialect, sql);
        let tokens = tokenizer.tokenize()?;
        Ok(DFParser {
            parser: Parser::new(
                tokens, dialect, // filename
            ),
            package_path : package_path,
            module_path : module_path,
        })
    }

    /// Parse a SQL statement and produce a set of statements with dialect
    pub fn parse_sql(sql: &str) -> Result<VecDeque<Statement>, ParserError> {
        let dialect = &GenericDialect {};
        DFParser::parse_sql_with_dialect(sql, dialect)
    }
    
    /// Parse a SQL statement and produce a set of statements with dialect
    pub fn parse_sql_with_scope(sql: &str,
        _filename: String,
        package_path: String, module_path:String) -> Result<VecDeque<Statement>, ParserError> {
        let dialect = &GenericDialect {};
        DFParser::parse_sql_with_dialect_and_scope(sql, dialect, _filename, package_path, module_path)
    }
    /// Parse a SQL statement and produce a set of statements
    pub fn parse_sql_with_dialect(
        sql: &str,
        dialect: &dyn Dialect,
    ) -> Result<VecDeque<Statement>, ParserError> {
        let parser = DFParser::new_with_dialect(sql, dialect)?;
        Self::parse_statements(parser)
    }
    /// Parse a SQL statement and produce a set of statements
    pub fn parse_sql_with_dialect_and_scope(
        sql: &str,
        dialect: &dyn Dialect,

        _filename: String,
        package_path: String,
        module_path: String,
  
    ) -> Result<VecDeque<Statement>, ParserError> {
        let parser = DFParser::new_with_dialect_and_scope(sql, dialect,_filename,package_path, module_path)?;
        Self::parse_statements(parser)
    }

    fn parse_statements(
        mut parser: DFParser,
    ) -> Result<VecDeque<Statement>, ParserError> {
        let mut stmts = VecDeque::new();
        let mut expecting_statement_delimiter = false;
        loop {
            // ignore empty statements (between successive statement delimiters)
            while parser.parser.consume_token(&Token::SemiColon) {
                expecting_statement_delimiter = false;
            }

            if parser.parser.peek_token() == Token::EOF {
                break;
            }
            if expecting_statement_delimiter {
                return parser.expected("end of statement", parser.parser.peek_token());
            }
            let result_statements = match parser.parser.next_token() {
                Token::Word(w) => match w.keyword {
                    Keyword::USE => {
                        Self::parse_use_from(&mut parser)},
                    _ => {
                        parser.parser.prev_token();
                        parser
                            .parse_statement()
                            .map(|stm| VecDeque::from(vec![stm]))
                    }
                },
                unexpected => parser.expected("end of statement", unexpected),
            };
            match result_statements {
                Ok(stms) => stmts.extend(stms),
                Err(err) => return Err(err),
            }

            expecting_statement_delimiter = true;
        }
        Ok(stmts)
    }

    /// Report unexpected token
    fn expected<T>(&self, expected: &str, found: Token) -> Result<T, ParserError> {
        parser_err!(format!("Expected {}, found: {}", expected, found))
    }

    fn parse_use_from(parser: &mut DFParser) -> Result<VecDeque<Statement>, ParserError> {
        // parser.expected("module identifier", parser.peek_token())?
        // TODO look into scopes...
        
        match parser.parser.next_token() {
             Token::SingleQuotedString(target_module_path)
                    
             => {
                // Note: in the GenericDialect SingleQuotedString is a string
                // we are staying in the same package

                let package_path = parser.package_path.clone();
                let package_name = basename(&package_path);

                // compute filename
                let target_module_name = basename(&target_module_path);
                let target_filename = sql_filename(&package_path, &target_module_path);

                // avoid duplicate uses
                if VISITED_FILES.lock().unwrap().contains(&target_filename) {
                    return Ok(VecDeque::new());
                }
                VISITED_FILES.lock().unwrap().insert(target_filename.clone());
                
                // create scopes
                let prefix = String::from("CREATE SCHEMA ")+ &package_name + "." + &target_module_name+";\n"; 
                
                // continue parsing  
                Self::parse_sql_file(&GenericDialect {}, target_filename, package_path.to_owned(),  target_module_path.to_owned(), prefix)

            }
            Token::Word(w) => {
                // Note: in the GenericDialect DoubleQuotedString is an
                // identifier -- for now that is not supported
                
                // parse
                let target_package_name =  w.value;
                let _ = parser.parser.expect_token(&Token::Period);
                let target_module_name = match parser.parser.parse_identifier(){
                    Ok(id) => id.value,
                    Err(_) => "".to_owned()
                };
                
                // check wither the new path is a package
                let old_package_path = parser.package_path.clone();
                let target_package_path = old_package_path.strip_suffix(&basename(&old_package_path)).unwrap().clone().to_owned()+&target_package_name;
                let target_root_file = target_package_path.clone()+"/"+&ROOT;
                if !Path::new(&target_root_file).is_file(){
                    return parser.expected("root target file", parser.parser.peek_token())
                } 
                // compute filename
                let target_module_path = target_module_name.clone();
                let target_filename = sql_filename(&target_package_path, &target_module_path);

                // avoid duplicate uses
                if VISITED_FILES.lock().unwrap().contains(&target_filename) {
                    return Ok(VecDeque::new());
                }
                VISITED_FILES.lock().unwrap().insert(target_filename.clone());

                // create scopes
                let prefix: String =
                    if VISITED_PACKAGES.lock().unwrap().contains(&target_package_name) {
                        String::new()   
                    } else {
                        VISITED_PACKAGES.lock().unwrap().insert(target_package_name.clone());
                        (String::from("CREATE DATABASE ")+ &target_package_name + ";\n").to_owned()
                    };
                let prefix = prefix + &String::from("CREATE SCHEMA ")+ &target_package_name + "." + &target_module_name+";\n";   
                
                // continue parsing
                Self::parse_sql_file(&GenericDialect {}, target_filename, target_package_path,  target_module_path, prefix)

            }
            unexpected => parser.expected("module identifier", unexpected)?,
        }
        // }
    }

    /// Parse a file of SQL statements and produce an Abstract Syntax Tree (AST)
    pub fn parse_sql_file(
        dialect: &dyn Dialect,
        filename: String,
        package_path: String,
        module_path: String,
        prefix: String
    ) -> Result<VecDeque<Statement>, ParserError> {
        let contents = fs::read_to_string(&filename)
            .unwrap_or_else(|_| panic!("Unable to read the file {}", &filename));
        let contents_with_prefix = prefix.clone() + &contents;   
        let parse_result = Self::tokenize_and_parse_sql(&*dialect, &contents_with_prefix, filename, package_path, module_path);

        parse_result
    }

    /// Tokenize and parse a SQL fragment and produce an Abstract Syntax Tree (AST)
    fn tokenize_and_parse_sql(
        dialect: &dyn Dialect,
        sql: &str,
        filename: String,
        package_path: String,
        module_path: String,
    ) -> Result<VecDeque<Statement>, ParserError> {
        let parser = match DFParser::new_with_dialect_and_scope(sql, dialect, filename, package_path, module_path) {
            Ok(it) => it,
            Err(err) => return Err(err),
        };
        Self::parse_statements(parser)
    }

    /// Parse a new expression
    pub fn parse_statement(&mut self) -> Result<Statement, ParserError> {
        match self.parser.peek_token() {
            Token::Word(w) => {
                match w.keyword {
                    Keyword::CREATE => {
                        // move one token forward
                        self.parser.next_token();
                        // use custom parsing
                        self.parse_create()
                    }
                    Keyword::DESCRIBE => {
                        // move one token forward
                        self.parser.next_token();
                        // use custom parsing
                        self.parse_describe()
                    }
                    _ => {
                        // use the native parser
                        let stm = self.parser.parse_statement()?;
                        // let stm = match stm {
                        //     SQLStatement::Query(query) => SQLStatement::CreateTable { temporary: true, name: ObjectName(vec![]), query: Some(query)},
                        //     s => s
                        // };
                        Ok(Statement::Statement(
                            Box::from(stm),
                            self.package_path.to_owned(),
                            self.module_path.to_owned()
                        ))
                    }
                }
            }
            _ => {
                // use the native parser
                let stm = self.parser.parse_statement()?;
                Ok(Statement::Statement(
                    Box::from(stm),
                    self.package_path.to_owned(),
                    self.module_path.to_owned()
                ))
            }
        }
    }

    pub fn parse_describe(&mut self) -> Result<Statement, ParserError> {
        let table_name = self.parser.parse_object_name()?;

        let des = DescribeTable {
            table_name: table_name.to_string(),
        };
        Ok(Statement::DescribeTable(
            des,
            self.package_path.to_owned(),
            self.module_path.to_owned()
        ))
    }

    /// Parse a SQL CREATE statement
    pub fn parse_create(&mut self) -> Result<Statement, ParserError> {
        if self.parser.parse_keyword(Keyword::EXTERNAL) {
            self.parse_create_external_table()
        } else {
            Ok(Statement::Statement(
                Box::from(self.parser.parse_create()?),
                self.package_path.to_owned(),
                self.module_path.to_owned()
            ))
        }
    }

    fn parse_partitions(&mut self) -> Result<Vec<String>, ParserError> {
        let mut partitions: Vec<String> = vec![];
        if !self.parser.consume_token(&Token::LParen)
            || self.parser.consume_token(&Token::RParen)
        {
            return Ok(partitions);
        }

        loop {
            if let Token::Word(_) = self.parser.peek_token() {
                let identifier = self.parser.parse_identifier()?;
                partitions.push(identifier.to_string());
            } else {
                return self.expected("partition name", self.parser.peek_token());
            }
            let comma = self.parser.consume_token(&Token::Comma);
            if self.parser.consume_token(&Token::RParen) {
                // allow a trailing comma, even though it's not in standard
                break;
            } else if !comma {
                return self.expected(
                    "',' or ')' after partition definition",
                    self.parser.peek_token(),
                );
            }
        }
        Ok(partitions)
    }

    // This is a copy of the equivalent implementation in sqlparser.
    fn parse_columns(
        &mut self,
    ) -> Result<(Vec<ColumnDef>, Vec<TableConstraint>), ParserError> {
        let mut columns = vec![];
        let mut constraints = vec![];
        if !self.parser.consume_token(&Token::LParen)
            || self.parser.consume_token(&Token::RParen)
        {
            return Ok((columns, constraints));
        }

        loop {
            if let Some(constraint) = self.parser.parse_optional_table_constraint()? {
                constraints.push(constraint);
            } else if let Token::Word(_) = self.parser.peek_token() {
                let column_def = self.parse_column_def()?;
                columns.push(column_def);
            } else {
                return self.expected(
                    "column name or constraint definition",
                    self.parser.peek_token(),
                );
            }
            let comma = self.parser.consume_token(&Token::Comma);
            if self.parser.consume_token(&Token::RParen) {
                // allow a trailing comma, even though it's not in standard
                break;
            } else if !comma {
                return self.expected(
                    "',' or ')' after column definition",
                    self.parser.peek_token(),
                );
            }
        }

        Ok((columns, constraints))
    }

    fn parse_column_def(&mut self) -> Result<ColumnDef, ParserError> {
        let name = self.parser.parse_identifier()?;
        let data_type = self.parser.parse_data_type()?;
        let collation = if self.parser.parse_keyword(Keyword::COLLATE) {
            Some(self.parser.parse_object_name()?)
        } else {
            None
        };
        let mut options = vec![];
        loop {
            if self.parser.parse_keyword(Keyword::CONSTRAINT) {
                let name = Some(self.parser.parse_identifier()?);
                if let Some(option) = self.parser.parse_optional_column_option()? {
                    options.push(ColumnOptionDef { name, option });
                } else {
                    return self.expected(
                        "constraint details after CONSTRAINT <name>",
                        self.parser.peek_token(),
                    );
                }
            } else if let Some(option) = self.parser.parse_optional_column_option()? {
                options.push(ColumnOptionDef { name: None, option });
            } else {
                break;
            };
        }
        Ok(ColumnDef {
            name,
            data_type,
            collation,
            options,
        })
    }

    fn parse_create_external_table(&mut self) -> Result<Statement, ParserError> {
        self.parser.expect_keyword(Keyword::TABLE)?;
        let if_not_exists =
            self.parser
                .parse_keywords(&[Keyword::IF, Keyword::NOT, Keyword::EXISTS]);
        let table_name = self.parser.parse_object_name()?;
        let (columns, _) = self.parse_columns()?;
        self.parser
            .expect_keywords(&[Keyword::STORED, Keyword::AS])?;

        // THIS is the main difference: we parse a different file format.
        let file_type = self.parse_file_format()?;

        let has_header = self.parse_csv_has_header();

        let has_delimiter = self.parse_has_delimiter();
        let delimiter = match has_delimiter {
            true => self.parse_delimiter()?,
            false => ',',
        };

        let file_compression_type = if self.parse_has_file_compression_type() {
            self.parse_file_compression_type()?
        } else {
            "".to_string()
        };

        let table_partition_cols = if self.parse_has_partition() {
            self.parse_partitions()?
        } else {
            vec![]
        };

        self.parser.expect_keyword(Keyword::LOCATION)?;
        let location = self.parser.parse_literal_string()?;
        
    
        // TODO Make consistent: Currently
        // - parser creates full name for external table, 
        // - planner does for normal tables.. 
        let current_package_name = basename(&self.package_path);
        let current_module_name = basename(&self.module_path);
        
        let enriched_table_name: ObjectName = match table_name{
            ObjectName(ids)=> {
                match ids.len(){
                    1 => {ObjectName(vec![Ident::new(current_package_name), Ident::new(current_module_name), ids[0].clone()])}
                    2 => {ObjectName(vec![Ident::new(current_package_name), ids[0].clone(), ids[1].clone()])}
                    _ => {ObjectName(ids)}
                }
            }
        };

        let create = CreateExternalTable {

            name: enriched_table_name.to_string(),
            columns,
            file_type,
            has_header,
            delimiter,
            location,
            table_partition_cols,
            if_not_exists,
            file_compression_type,
        };
        Ok(Statement::CreateExternalTable(
            create,
            self.package_path.to_owned(), self.module_path.to_owned()
        ))
    }

    /// Parses the set of valid formats
    fn parse_file_format(&mut self) -> Result<String, ParserError> {
        match self.parser.next_token() {
            Token::Word(w) => parse_file_type(&w.value),
            unexpected => self.expected("one of PARQUET, NDJSON, or CSV", unexpected),
        }
    }

    /// Parses the set of
    fn parse_file_compression_type(&mut self) -> Result<String, ParserError> {
        match self.parser.next_token() {
            Token::Word(w) => parse_file_compression_type(&w.value),
            unexpected => self.expected("one of GZIP, BZIP2", unexpected),
        }
    }

    fn consume_token(&mut self, expected: &Token) -> bool {
        let token = self.parser.peek_token().to_string().to_uppercase();
        let token = Token::make_keyword(&token);
        if token == *expected {
            self.parser.next_token();
            true
        } else {
            false
        }
    }
    fn parse_has_file_compression_type(&mut self) -> bool {
        self.consume_token(&Token::make_keyword("COMPRESSION"))
            & self.consume_token(&Token::make_keyword("TYPE"))
    }

    fn parse_csv_has_header(&mut self) -> bool {
        self.consume_token(&Token::make_keyword("WITH"))
            & self.consume_token(&Token::make_keyword("HEADER"))
            & self.consume_token(&Token::make_keyword("ROW"))
    }

    fn parse_has_delimiter(&mut self) -> bool {
        self.consume_token(&Token::make_keyword("DELIMITER"))
    }

    fn parse_delimiter(&mut self) -> Result<char, ParserError> {
        let token = self.parser.parse_literal_string()?;
        match token.len() {
            1 => Ok(token.chars().next().unwrap()),
            _ => Err(ParserError::TokenizerError(
                "Delimiter must be a single char".to_string(),
            )),
        }
    }

    fn parse_has_partition(&mut self) -> bool {
        self.consume_token(&Token::make_keyword("PARTITIONED"))
            & self.consume_token(&Token::make_keyword("BY"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sqlparser::ast::{DataType, Ident};

    fn expect_parse_ok(sql: &str, expected: Statement) -> Result<(), ParserError> {
        let statements = DFParser::parse_sql(sql)?;
        assert_eq!(
            statements.len(),
            1,
            "Expected to parse exactly one statement"
        );
        assert_eq!(statements[0], expected);
        Ok(())
    }

    /// Parses sql and asserts that the expected error message was found
    fn expect_parse_error(sql: &str, expected_error: &str) {
        match DFParser::parse_sql(sql) {
            Ok(statements) => {
                panic!(
                    "Expected parse error for '{}', but was successful: {:?}",
                    sql, statements
                );
            }
            Err(e) => {
                let error_message = e.to_string();
                assert!(
                    error_message.contains(expected_error),
                    "Expected error '{}' not found in actual error '{}'",
                    expected_error,
                    error_message
                );
            }
        }
    }

    fn make_column_def(name: impl Into<String>, data_type: DataType) -> ColumnDef {
        ColumnDef {
            name: Ident {
                value: name.into(),
                quote_style: None,
            },
            data_type,
            collation: None,
            options: vec![],
        }
    }

    #[test]
    fn create_external_table() -> Result<(), ParserError> {
        // positive case
        let sql = "CREATE EXTERNAL TABLE t(c1 int) STORED AS CSV LOCATION 'foo.csv'";
        let display = None;
        let expected = Statement::CreateExternalTable(
            CreateExternalTable {
                name: "t".into(),
                columns: vec![make_column_def("c1", DataType::Int(display))],
                file_type: "CSV".to_string(),
                has_header: false,
                delimiter: ',',
                location: "foo.csv".into(),
                table_partition_cols: vec![],
                if_not_exists: false,
                file_compression_type: "".to_string(),
            },
            String::new(), String::new()
        );
        expect_parse_ok(sql, expected)?;

        // positive case with delimiter
        let sql = "CREATE EXTERNAL TABLE t(c1 int) STORED AS CSV DELIMITER '|' LOCATION 'foo.csv'";
        let display = None;
        let expected = Statement::CreateExternalTable(
            CreateExternalTable {
                name: "t".into(),
                columns: vec![make_column_def("c1", DataType::Int(display))],
                file_type: "CSV".to_string(),
                has_header: false,
                delimiter: '|',
                location: "foo.csv".into(),
                table_partition_cols: vec![],
                if_not_exists: false,
                file_compression_type: "".to_string(),
            },
            String::new(), String::new()
        );
        expect_parse_ok(sql, expected)?;

        // positive case: partitioned by
        let sql = "CREATE EXTERNAL TABLE t(c1 int) STORED AS CSV PARTITIONED BY (p1, p2) LOCATION 'foo.csv'";
        let display = None;
        let expected = Statement::CreateExternalTable(
            CreateExternalTable {
                name: "t".into(),
                columns: vec![make_column_def("c1", DataType::Int(display))],
                file_type: "CSV".to_string(),
                has_header: false,
                delimiter: ',',
                location: "foo.csv".into(),
                table_partition_cols: vec!["p1".to_string(), "p2".to_string()],
                if_not_exists: false,
                file_compression_type: "".to_string(),
            },
            String::new(), String::new()
        );
        expect_parse_ok(sql, expected)?;

        // positive case: it is ok for case insensitive sql stmt with `WITH HEADER ROW` tokens
        let sqls = vec![
            "CREATE EXTERNAL TABLE t(c1 int) STORED AS CSV WITH HEADER ROW LOCATION 'foo.csv'",
            "CREATE EXTERNAL TABLE t(c1 int) STORED AS CSV with header row LOCATION 'foo.csv'"
        ];
        for sql in sqls {
            let expected = Statement::CreateExternalTable(
                CreateExternalTable {
                    name: "t".into(),
                    columns: vec![make_column_def("c1", DataType::Int(display))],
                    file_type: "CSV".to_string(),
                    has_header: true,
                    delimiter: ',',
                    location: "foo.csv".into(),
                    table_partition_cols: vec![],
                    if_not_exists: false,
                    file_compression_type: "".to_string(),
                },
                String::new(), String::new()
            );
            expect_parse_ok(sql, expected)?;
        }

        // positive case: it is ok for sql stmt with `COMPRESSION TYPE GZIP` tokens
        let sqls = vec![
            ("CREATE EXTERNAL TABLE t(c1 int) STORED AS CSV COMPRESSION TYPE GZIP LOCATION 'foo.csv'", "GZIP"),
            ("CREATE EXTERNAL TABLE t(c1 int) STORED AS CSV COMPRESSION TYPE BZIP2 LOCATION 'foo.csv'", "BZIP2"),
        ];
        for (sql, file_compression_type) in sqls {
            let expected = Statement::CreateExternalTable(
                CreateExternalTable {
                    name: "t".into(),
                    columns: vec![make_column_def("c1", DataType::Int(display))],
                    file_type: "CSV".to_string(),
                    has_header: false,
                    delimiter: ',',
                    location: "foo.csv".into(),
                    table_partition_cols: vec![],
                    if_not_exists: false,
                    file_compression_type: file_compression_type.to_owned(),
                },
                String::new(), String::new()
            );
            expect_parse_ok(sql, expected)?;
        }

        // positive case: it is ok for parquet files not to have columns specified
        let sql = "CREATE EXTERNAL TABLE t STORED AS PARQUET LOCATION 'foo.parquet'";
        let expected = Statement::CreateExternalTable(
            CreateExternalTable {
                name: "t".into(),
                columns: vec![],
                file_type: "PARQUET".to_string(),
                has_header: false,
                delimiter: ',',
                location: "foo.parquet".into(),
                table_partition_cols: vec![],
                if_not_exists: false,
                file_compression_type: "".to_string(),
            },
            String::new(), String::new()
        );
        expect_parse_ok(sql, expected)?;

        // positive case: it is ok for parquet files to be other than upper case
        let sql = "CREATE EXTERNAL TABLE t STORED AS parqueT LOCATION 'foo.parquet'";
        let expected = Statement::CreateExternalTable(
            CreateExternalTable {
                name: "t".into(),
                columns: vec![],
                file_type: "PARQUET".to_string(),
                has_header: false,
                delimiter: ',',
                location: "foo.parquet".into(),
                table_partition_cols: vec![],
                if_not_exists: false,
                file_compression_type: "".to_string(),
            },
            String::new(), String::new()
        );
        expect_parse_ok(sql, expected)?;

        // positive case: it is ok for avro files not to have columns specified
        let sql = "CREATE EXTERNAL TABLE t STORED AS AVRO LOCATION 'foo.avro'";
        let expected = Statement::CreateExternalTable(
            CreateExternalTable {
                name: "t".into(),
                columns: vec![],
                file_type: "AVRO".to_string(),
                has_header: false,
                delimiter: ',',
                location: "foo.avro".into(),
                table_partition_cols: vec![],
                if_not_exists: false,
                file_compression_type: "".to_string(),
            },
            String::new(), String::new()
        );
        expect_parse_ok(sql, expected)?;

        // positive case: it is ok for avro files not to have columns specified
        let sql =
            "CREATE EXTERNAL TABLE IF NOT EXISTS t STORED AS PARQUET LOCATION 'foo.parquet'";
        let expected = Statement::CreateExternalTable(
            CreateExternalTable {
                name: "t".into(),
                columns: vec![],
                file_type: "PARQUET".to_string(),
                has_header: false,
                delimiter: ',',
                location: "foo.parquet".into(),
                table_partition_cols: vec![],
                if_not_exists: true,
                file_compression_type: "".to_string(),
            },
            String::new(), String::new()
        );
        expect_parse_ok(sql, expected)?;

        // Error cases: partition column does not support type
        let sql =
            "CREATE EXTERNAL TABLE t(c1 int) STORED AS CSV PARTITIONED BY (p1 int) LOCATION 'foo.csv'";
        expect_parse_error(sql, "sql parser error: Expected ',' or ')' after partition definition, found: int");

        Ok(())
    }
}
