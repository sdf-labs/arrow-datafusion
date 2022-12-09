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
    ast::{
        ColumnDef, ColumnOptionDef, HiveDistributionStyle, Ident, ObjectName,
        Statement as SQLStatement, TableConstraint,
    },
    dialect::{keywords::Keyword, Dialect, GenericDialect},
    parser::{Parser, ParserError},
    tokenizer::{Token, Tokenizer},
};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    fmt, fs,
    path::{Path, PathBuf},
    process::exit,
};
extern crate regex;

use lazy_static::lazy_static;
use std::sync::Mutex;
// use crate::{dialect::Dialect, parser::{Parser, ParserError}, ast::Statement, tokenizer::Token, keywords::Keyword};

use once_cell::sync::OnceCell;

pub static VERBOSE_FLAG: OnceCell<i8> = OnceCell::new();

lazy_static! {
    /// collects all files that have been visited so far
    pub static ref VISITED_FILES: Mutex<HashSet<String>> = Mutex::new(HashSet::new());
    // collects all packages that have been visited so far
    pub static ref VISITED_CATALOGS: Mutex<HashSet<String>> = Mutex::new(HashSet::new());
    // collects all external table locations, catalog.schema.table -> relative
    pub static ref VISITED_SCHEMAS: Mutex<HashSet<String>> = Mutex::new(HashSet::new());
    // collects all external table locations, catalog.schema.table -> relative path
    pub static ref LOCATIONS: Mutex<HashSet<String>> = Mutex::new(HashSet::new());
}
pub static CATALOG: &str = "catalog.yml";
pub static WORKSPACE: &str = "workspace.yml";
pub static DATA_DIR: &str = ".data";
pub const SOURCE_CACHE: &str = "source_cache.csv";
pub const DATA_CACHE: &str = "data_cache.csv";

pub fn visit(filename: &str, catalog: &str, schema: &str) {
    VISITED_FILES.lock().unwrap().insert(filename.to_owned());
    VISITED_CATALOGS.lock().unwrap().insert(catalog.to_owned());
    VISITED_SCHEMAS
        .lock()
        .unwrap()
        .insert(format!("{}.{}", catalog, schema));
}

// Removes directory path and returns the file name; like path.filename, but for strings
pub fn basename(path: &str) -> String {
    match path.rfind('/') {
        Some(i) => path[i + 1..].to_owned(),
        None => path.to_owned(),
    }
}
// Removes basename from directory path
pub fn parent(path: &str) -> String {
    match path.rfind('/') {
        Some(i) => path[0..i].to_owned(),
        None => "".to_owned(),
    }
}
pub fn extension(path: &str) -> String {
    match path.rfind('.') {
        Some(i) => path[i + 1..].to_owned(),
        None => "".to_owned(),
    }
}

pub fn strip_extension(path: &str, ext: &str) -> String {
    match path.strip_suffix(ext) {
        Some(base) => base.to_owned(),
        None => path.to_owned(),
    }
}

pub fn find_package_file(starting_directory: &Path) -> Option<PathBuf> {
    let mut path: PathBuf = starting_directory.into();
    let root_filename = Path::new(CATALOG);

    loop {
        path.push(root_filename);
        if path.is_file() {
            break Some(path.canonicalize().unwrap());
        }
        if !(path.pop() && path.pop()) {
            // remove file && remove parent
            break None;
        }
    }
}

pub fn find_package_path(starting_directory: &Path) -> Option<PathBuf> {
    if let Some(path) = find_package_file(Path::new(&starting_directory)) {
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
    /// File compression type (GZIP, BZIP2, XZ)
    pub file_compression_type: String,
    /// Table(provider) specific options
    pub options: HashMap<String, String>,
}

impl fmt::Display for CreateExternalTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CREATE EXTERNAL TABLE ")?;
        if self.if_not_exists {
            write!(f, "IF NOT EXISTS ")?;
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
    pub table_name: ObjectName,
}

/// DataFusion Statement representations.
///
/// Tokens parsed by `DFParser` are converted into these values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Statement {
    /// ANSI SQL AST node with package_schema_path
    Statement(Box<SQLStatement>),
    /// Extension: `CREATE EXTERNAL TABLE` with package_path module_path
    CreateExternalTable(CreateExternalTable),
    /// Extension: `DESCRIBE TABLE` with package_path module_path
    DescribeTable(DescribeTable),
}

/// SDF StatementMeta
///
/// The location at which the statement is defined.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatementMeta {
    pub catalog: String,
    pub schema: String,
    pub table: String,
    pub line_number: i32,
    pub filename: String,
}

impl StatementMeta {
    /// An empty statement definition location
    pub fn empty() -> Self {
        StatementMeta {
            catalog: String::new(),
            schema: String::new(),
            table: String::new(),
            line_number: 0,
            filename: String::new(),
        }
    }
    /// An statement definition location without line number
    //   That's' what Datafusion gives us today
    pub fn new(catalog: String, schema: String) -> Self {
        StatementMeta {
            catalog,
            schema,
            table: String::new(),
            line_number: 0,
            filename: String::new(),
        }
    }
    /// An statement definition location without line number
    pub fn new_with_table(
        catalog: String,
        schema: String,
        table: String,
        filename: String,
    ) -> Self {
        StatementMeta {
            catalog,
            schema,
            table,
            line_number: 0,
            filename,
        }
    }
    /// Return schema_file name, which is relative to workspace
    pub fn schema_filename(&self) -> String {
        format!("{},{}.sql", self.catalog, self.schema)
    }
}

/// SQL Parser
#[allow(dead_code)]
pub struct DFParser<'a> {
    parser: Parser<'a>,
    catalog: String,
    schema: String,
    table: String,
    filename: String,
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
            catalog: String::new(),
            schema: String::new(),
            table: String::new(),
            filename: String::new(),
        })
    }

    pub fn new_with_dialect_and_scope(
        sql: &str,
        dialect: &'a dyn Dialect,
        filename: String,
        catalog: String,
        schema: String,
        table: String,
    ) -> Result<Self, ParserError> {
        let mut tokenizer = Tokenizer::new(dialect, sql);
        let tokens = tokenizer.tokenize()?;
        Ok(DFParser {
            parser: Parser::new(
                tokens, dialect, // filename
            ),
            catalog,
            schema,
            table,
            filename,
        })
    }

    /// Parse a SQL statement and produce a set of statements with dialect
    pub fn parse_sql(sql: &str) -> Result<VecDeque<Statement>, ParserError> {
        let dialect = &GenericDialect {};
        DFParser::parse_sql_with_dialect(sql, dialect)
    }

    /// Parse a SQL statement and produce a set of statements with dialect
    pub fn parse_sql_with_scope(
        sql: &str,
        filename: String,
        catalog: String,
        schema: String,
        table: String,
    ) -> Result<VecDeque<(Statement, StatementMeta)>, ParserError> {
        let dialect = &GenericDialect {};
        DFParser::parse_sql_with_dialect_and_scope(
            sql, dialect, filename, catalog, schema, table,
        )
    }
    /// Parse a SQL statement and produce a set of statements
    pub fn parse_sql_with_dialect(
        sql: &str,
        dialect: &dyn Dialect,
    ) -> Result<VecDeque<Statement>, ParserError> {
        let parser = DFParser::new_with_dialect(sql, dialect)?;
        Self::parse_statements(parser)
            .map(|list| list.into_iter().map(|elem| elem.0).collect())
    }

    /// Parse a SQL statement and produce a set of statements inside a given scope
    pub fn parse_sql_with_dialect_and_scope(
        sql: &str,
        dialect: &dyn Dialect,

        filename: String,
        catalog: String,
        schema: String,
        table: String,
    ) -> Result<VecDeque<(Statement, StatementMeta)>, ParserError> {
        let parser = DFParser::new_with_dialect_and_scope(
            sql,
            dialect,
            filename.to_owned(),
            catalog,
            schema,
            table,
        )?;
        match Self::parse_statements(parser) {
            Ok(res) => Ok(res),
            Err(err) => {
                error!("{}: {}", &filename, err);
                exit(1)
            }
        }
    }

    fn parse_statements(
        mut parser: DFParser,
    ) -> Result<VecDeque<(Statement, StatementMeta)>, ParserError> {
        let mut stmts: VecDeque<(Statement, StatementMeta)> = VecDeque::new();
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
                return parser.expected("End of statement", parser.parser.peek_token());
            }
            let result_statements = match parser.parser.next_token() {
                Token::Word(w) => match w.keyword {
                    Keyword::USE => Self::parse_use(&mut parser),
                    _ => {
                        parser.parser.prev_token();
                        parser.parse_statement().map(|op| VecDeque::from([op]))
                    }
                },
                unexpected => parser.expected("End of statement", unexpected),
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

    fn parse_use(
        parser: &mut DFParser,
    ) -> Result<VecDeque<(Statement, StatementMeta)>, ParserError> {
        let next = parser.parser.next_token();
        match next.clone() {
            // Token::SingleQuotedString(schema_path) => {
            //     // we are staying in the same catalog -- we are reusing parser.catalog

            //     // compute filename
            //     let schema = basename(&schema_path);
            //     let schema_filename = format!(
            //         "{}/{}/{}.sql",
            //         parser.workspace_path, parser.catalog, schema_path
            //     );

            //     // avoid duplicate uses
            //     if VISITED_FILES.lock().unwrap().contains(&schema_filename) {
            //         return Ok(VecDeque::new());
            //     }
            //     VISITED_FILES
            //         .lock()
            //         .unwrap()
            //         .insert(schema_filename.clone());

            //     let regex = Regex::new(r"^[/a-z0-9_]*$").unwrap();
            //     if !regex.is_match(&schema_path) {
            //         return parser.wrong_use(&format!("Schema path must consist only of lowercase chars, digits or '_' separated by '/', found {}",next), next );
            //     }
            //     if !Path::new(&schema_filename).is_file() {
            //         return parser.wrong_use(
            //             &format!("Missing schema file {}", schema_filename),
            //             next,
            //         );
            //     };

            //     // create scopes
            //     let created_schema =
            //         format!("CREATE SCHEMA {}.{};\n", &parser.catalog, &schema);

            //     // continue parsing
            //     Self::parse_sql_file(
            //         &GenericDialect {},
            //         schema_filename,
            //         parser.catalog.to_owned(),
            //         schema_path.to_owned(),
            //         created_schema,
            //         workspace_path,
            //     )
            // }
            Token::Word(w) => {
                // switch to a possibly new catalog

                //parse
                let catalog = w.value.clone();
                let _ = parser.parser.expect_token(&Token::Period);
                let schema = match parser.parser.parse_identifier() {
                    Ok(id) => id.value,
                    Err(_) => "".to_owned(),
                };
                let _ = parser.parser.expect_token(&Token::Period);
                let table = match parser.parser.parse_identifier() {
                    Ok(id) => id.value,
                    Err(_) => "".to_owned(),
                };
                // check catalog/schema naming
                // let regex = Regex::new(r"^[a-z0-9_]*$").unwrap();
                // if !regex.is_match(&catalog)
                //     || !regex.is_match(&schema)
                //     || !regex.is_match(&table)
                // {
                //     return parser.wrong_use(&format!("Catalog, schema, and table names must only be lowercase, digits or '_', found {}",catalog), next );
                // }

                if parser.catalog == "" {
                    println!(
                        "Source not under workspace -- skipping 'use {}.{}.{}' statement",
                        catalog, schema, table
                    );
                    return Ok(VecDeque::new());
                }

                // check whether new catalog exists
                let schema_filename = format!("{}/{}.sql", catalog, schema);
                let table_filename = format!("{}/{}/{}.sql", catalog, schema, table);
                let (is_table, filename) = if Path::new(&table_filename).is_file() {
                    (true, table_filename)
                } else if Path::new(&schema_filename).is_file() {
                    (false, schema_filename)
                } else {
                    return Err(ParserError::ParserError(
                        format!(
                            "Missing schema file {} or table file {} ",
                            schema_filename, table_filename
                        )
                        .to_owned(),
                    ));
                };
                info!("-- USE {}.{}.{} from {}", catalog, schema, table, filename);
                // avoid duplicate uses
                if VISITED_FILES.lock().unwrap().contains(&filename) {
                    return Ok(VecDeque::new());
                }
                VISITED_FILES.lock().unwrap().insert(filename.to_owned());

                // create scopes
                let mut created_catalog = String::new();
                let mut created_schema = String::new();
                if !VISITED_CATALOGS.lock().unwrap().contains(&catalog) {
                    VISITED_CATALOGS.lock().unwrap().insert(catalog.to_owned());
                    created_catalog = format!("CREATE DATABASE {};\n", &catalog);
                    info!("{}", created_catalog);
                };
                let schema_id = format!("{}.{}", catalog, schema);
                if !VISITED_SCHEMAS.lock().unwrap().contains(&schema_id) {
                    VISITED_SCHEMAS.lock().unwrap().insert(schema_id);
                    created_schema = format!("CREATE SCHEMA {}.{};\n", &catalog, &schema);
                    info!("{}", created_schema);
                };

                // continue parsing
                Self::parse_sql_file(
                    &GenericDialect {},
                    filename,
                    catalog,
                    schema,
                    if is_table { table } else { String::new() },
                    created_catalog + &created_schema,
                )
            }
            unexpected => parser.expected("Object identifier", unexpected)?,
        }
        // }
    }

    /// Parse a file of SQL statements and produce an Abstract Syntax Tree (AST)
    pub fn parse_sql_file(
        dialect: &dyn Dialect,
        filename: String,
        catalog: String,
        schema: String,
        table: String,
        prefix: String,
    ) -> Result<VecDeque<(Statement, StatementMeta)>, ParserError> {
        let contents = fs::read_to_string(&filename)
            .unwrap_or_else(|_| panic!("Unable to read the file {}", &filename));
        let contents_with_prefix = prefix.clone() + &contents;

        let dialect: &dyn Dialect = &*dialect;
        let sql: &str = &contents_with_prefix;
        let parser = DFParser::new_with_dialect_and_scope(
            sql,
            dialect,
            filename.to_owned(),
            catalog,
            schema,
            table,
        )?;
        match Self::parse_statements(parser) {
            Ok(res) => Ok(res),
            Err(err) => {
                error!("{}: {}", &filename, err);
                exit(1)
            }
        }
    }

    /// Parse a new expression
    pub fn parse_statement(&mut self) -> Result<(Statement, StatementMeta), ParserError> {
        let token: Token = self.parser.peek_token();
        // let line_number = token.
        match token {
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
                    Keyword::SELECT | Keyword::WITH | Keyword::VALUES => {
                        // self.parser.prev_token();
                        let base_query = self.parser.parse_query()?;
                        let boxed_query = Box::new(base_query.to_owned());
                        // println!("SELECT:: {} ; catalog {} schema {} table {}", base_query, self.catalog, self.schema, self.table);
                        if self.table != "" {
                            // this is a select of of table definition
                            let c = Ident::new(&self.catalog);
                            let s = Ident::new(&self.schema);
                            let t = Ident::new(&self.table);
                            let create_table_statement =
                                sqlparser::ast::Statement::CreateTable {
                                    or_replace: false,
                                    temporary: false,
                                    external: false,
                                    global: None,
                                    if_not_exists: false,
                                    /// Table name
                                    name: ObjectName(vec![c, s, t]), // vec of ident
                                    /// Optional schema
                                    columns: vec![],
                                    constraints: vec![],
                                    hive_distribution: HiveDistributionStyle::NONE,
                                    hive_formats: None,
                                    table_properties: vec![],
                                    with_options: vec![],
                                    file_format: None,
                                    location: None,
                                    query: Some(boxed_query),
                                    without_rowid: false,
                                    like: None,
                                    clone: None,
                                    engine: None,
                                    default_charset: None,
                                    collation: None,
                                    on_commit: None,
                                    /// Click house "ON CLUSTER" clause:
                                    /// <https://clickhouse.com/docs/en/sql-reference/distributed-ddl/>
                                    on_cluster: None,
                                };
                            Ok((
                                Statement::Statement(Box::from(create_table_statement)),
                                self.with_meta("".to_owned()),
                            ))
                        } else {
                            // a usual select
                            let query_statement =
                                sqlparser::ast::Statement::Query(boxed_query);
                            Ok((
                                Statement::Statement(Box::from(query_statement)),
                                self.with_meta("".to_owned()),
                            ))
                        }
                    }
                    _ => {
                        let stm = self.parser.parse_statement()?;
                        Ok((
                            Statement::Statement(Box::from(stm)),
                            self.with_meta("".to_owned()),
                        ))
                    }
                }
            }
            _ => {
                // use the native parser
                let stm = self.parser.parse_statement()?;
                Ok((
                    Statement::Statement(Box::from(stm)),
                    self.with_meta("".to_owned()),
                ))
            }
        }
    }

    pub fn parse_describe(&mut self) -> Result<(Statement, StatementMeta), ParserError> {
        let table_name = self.parser.parse_object_name()?;
        let table_string = table_name.to_owned();
        let des = DescribeTable {
            table_name: table_name,
        };
        Ok((
            Statement::DescribeTable(des),
            self.with_meta(table_string.to_string()),
        ))
    }

    /// Parse a SQL CREATE statement
    pub fn parse_create(&mut self) -> Result<(Statement, StatementMeta), ParserError> {
        if self.parser.parse_keyword(Keyword::EXTERNAL) {
            self.parse_create_external_table()
        } else {
            let stm = self.parser.parse_create()?;
            let table = match &stm {
                SQLStatement::CreateView { name, .. }
                | SQLStatement::CreateTable { name, .. }
                | SQLStatement::CreateVirtualTable { name, .. } => name.to_owned(),
                _ => sqlparser::ast::ObjectName(vec![]),
            };
            Ok((
                Statement::Statement(Box::from(stm)),
                self.with_meta(table.to_string()),
            ))
        }
    }

    fn with_meta(&mut self, table: String) -> StatementMeta {
        StatementMeta::new_with_table(
            self.catalog.to_owned(),
            self.schema.to_owned(),
            table,
            self.filename.to_owned(),
        )
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

    fn parse_create_external_table(
        &mut self,
    ) -> Result<(Statement, StatementMeta), ParserError> {
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

        let options = if self.parse_has_options() {
            self.parse_options()?
        } else {
            HashMap::new()
        };

        self.parser.expect_keyword(Keyword::LOCATION)?;
        let location = self.parser.parse_literal_string()?;
        let location2 = location.to_owned();
        let file_type2 = file_type.to_owned();
        let create = CreateExternalTable {
            name: table_name.to_string(),
            columns,
            file_type,
            has_header,
            delimiter,
            location,
            table_partition_cols,
            if_not_exists,
            file_compression_type,
            options,
        };

        LOCATIONS.lock().unwrap().insert(format!(
            "{}::{}",
            location2.to_ascii_lowercase(),
            file_type2.to_ascii_lowercase()
        ));

        Ok((
            Statement::CreateExternalTable(create),
            self.with_meta(table_name.to_string().to_owned()),
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
            unexpected => self.expected("one of GZIP, BZIP2, XZ", unexpected),
        }
    }

    fn parse_has_options(&mut self) -> bool {
        self.consume_token(&Token::make_keyword("OPTIONS"))
    }

    //
    fn parse_options(&mut self) -> Result<HashMap<String, String>, ParserError> {
        let mut options: HashMap<String, String> = HashMap::new();
        self.parser.expect_token(&Token::LParen)?;

        loop {
            let key = self.parser.parse_literal_string()?;
            let value = self.parser.parse_literal_string()?;
            options.insert(key.to_string(), value.to_string());
            let comma = self.parser.consume_token(&Token::Comma);
            if self.parser.consume_token(&Token::RParen) {
                // allow a trailing comma, even though it's not in standard
                break;
            } else if !comma {
                return self.expected(
                    "',' or ')' after option definition",
                    self.parser.peek_token(),
                );
            }
        }
        Ok(options)
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
        self.parser
            .parse_keywords(&[Keyword::WITH, Keyword::HEADER, Keyword::ROW])
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
        self.parser
            .parse_keywords(&[Keyword::PARTITIONED, Keyword::BY])
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
        let expected = Statement::CreateExternalTable(CreateExternalTable {
            name: "t".into(),
            columns: vec![make_column_def("c1", DataType::Int(display))],
            file_type: "CSV".to_string(),
            has_header: false,
            delimiter: ',',
            location: "foo.csv".into(),
            table_partition_cols: vec![],
            if_not_exists: false,
            file_compression_type: "".to_string(),
            options: HashMap::new(),
        });
        expect_parse_ok(sql, expected)?;

        // positive case with delimiter
        let sql = "CREATE EXTERNAL TABLE t(c1 int) STORED AS CSV DELIMITER '|' LOCATION 'foo.csv'";
        let display = None;
        let expected = Statement::CreateExternalTable(CreateExternalTable {
            name: "t".into(),
            columns: vec![make_column_def("c1", DataType::Int(display))],
            file_type: "CSV".to_string(),
            has_header: false,
            delimiter: '|',
            location: "foo.csv".into(),
            table_partition_cols: vec![],
            if_not_exists: false,
            file_compression_type: "".to_string(),
            options: HashMap::new(),
        });
        expect_parse_ok(sql, expected)?;

        // positive case: partitioned by
        let sql = "CREATE EXTERNAL TABLE t(c1 int) STORED AS CSV PARTITIONED BY (p1, p2) LOCATION 'foo.csv'";
        let display = None;
        let expected = Statement::CreateExternalTable(CreateExternalTable {
            name: "t".into(),
            columns: vec![make_column_def("c1", DataType::Int(display))],
            file_type: "CSV".to_string(),
            has_header: false,
            delimiter: ',',
            location: "foo.csv".into(),
            table_partition_cols: vec!["p1".to_string(), "p2".to_string()],
            if_not_exists: false,
            file_compression_type: "".to_string(),
            options: HashMap::new(),
        });
        expect_parse_ok(sql, expected)?;

        // positive case: it is ok for case insensitive sql stmt with `WITH HEADER ROW` tokens
        let sqls = vec![
            "CREATE EXTERNAL TABLE t(c1 int) STORED AS CSV WITH HEADER ROW LOCATION 'foo.csv'",
            "CREATE EXTERNAL TABLE t(c1 int) STORED AS CSV with header row LOCATION 'foo.csv'"
        ];
        for sql in sqls {
            let expected = Statement::CreateExternalTable(CreateExternalTable {
                name: "t".into(),
                columns: vec![make_column_def("c1", DataType::Int(display))],
                file_type: "CSV".to_string(),
                has_header: true,
                delimiter: ',',
                location: "foo.csv".into(),
                table_partition_cols: vec![],
                if_not_exists: false,
                file_compression_type: "".to_string(),
                options: HashMap::new(),
            });
            expect_parse_ok(sql, expected)?;
        }

        // positive case: it is ok for sql stmt with `COMPRESSION TYPE GZIP` tokens
        let sqls = vec![
            ("CREATE EXTERNAL TABLE t(c1 int) STORED AS CSV COMPRESSION TYPE GZIP LOCATION 'foo.csv'", "GZIP"),
            ("CREATE EXTERNAL TABLE t(c1 int) STORED AS CSV COMPRESSION TYPE BZIP2 LOCATION 'foo.csv'", "BZIP2"),
            ("CREATE EXTERNAL TABLE t(c1 int) STORED AS CSV COMPRESSION TYPE XZ LOCATION 'foo.csv'", "XZ"),
        ];
        for (sql, file_compression_type) in sqls {
            let expected = Statement::CreateExternalTable(CreateExternalTable {
                name: "t".into(),
                columns: vec![make_column_def("c1", DataType::Int(display))],
                file_type: "CSV".to_string(),
                has_header: false,
                delimiter: ',',
                location: "foo.csv".into(),
                table_partition_cols: vec![],
                if_not_exists: false,
                file_compression_type: file_compression_type.to_owned(),
                options: HashMap::new(),
            });
            expect_parse_ok(sql, expected)?;
        }

        // positive case: it is ok for parquet files not to have columns specified
        let sql = "CREATE EXTERNAL TABLE t STORED AS PARQUET LOCATION 'foo.parquet'";
        let expected = Statement::CreateExternalTable(CreateExternalTable {
            name: "t".into(),
            columns: vec![],
            file_type: "PARQUET".to_string(),
            has_header: false,
            delimiter: ',',
            location: "foo.parquet".into(),
            table_partition_cols: vec![],
            if_not_exists: false,
            file_compression_type: "".to_string(),
            options: HashMap::new(),
        });
        expect_parse_ok(sql, expected)?;

        // positive case: it is ok for parquet files to be other than upper case
        let sql = "CREATE EXTERNAL TABLE t STORED AS parqueT LOCATION 'foo.parquet'";
        let expected = Statement::CreateExternalTable(CreateExternalTable {
            name: "t".into(),
            columns: vec![],
            file_type: "PARQUET".to_string(),
            has_header: false,
            delimiter: ',',
            location: "foo.parquet".into(),
            table_partition_cols: vec![],
            if_not_exists: false,
            file_compression_type: "".to_string(),
            options: HashMap::new(),
        });
        expect_parse_ok(sql, expected)?;

        // positive case: it is ok for avro files not to have columns specified
        let sql = "CREATE EXTERNAL TABLE t STORED AS AVRO LOCATION 'foo.avro'";
        let expected = Statement::CreateExternalTable(CreateExternalTable {
            name: "t".into(),
            columns: vec![],
            file_type: "AVRO".to_string(),
            has_header: false,
            delimiter: ',',
            location: "foo.avro".into(),
            table_partition_cols: vec![],
            if_not_exists: false,
            file_compression_type: "".to_string(),
            options: HashMap::new(),
        });
        expect_parse_ok(sql, expected)?;

        // positive case: it is ok for avro files not to have columns specified
        let sql =
            "CREATE EXTERNAL TABLE IF NOT EXISTS t STORED AS PARQUET LOCATION 'foo.parquet'";
        let expected = Statement::CreateExternalTable(CreateExternalTable {
            name: "t".into(),
            columns: vec![],
            file_type: "PARQUET".to_string(),
            has_header: false,
            delimiter: ',',
            location: "foo.parquet".into(),
            table_partition_cols: vec![],
            if_not_exists: true,
            file_compression_type: "".to_string(),
            options: HashMap::new(),
        });
        expect_parse_ok(sql, expected)?;

        // Error cases: partition column does not support type
        let sql =
            "CREATE EXTERNAL TABLE t(c1 int) STORED AS CSV PARTITIONED BY (p1 int) LOCATION 'foo.csv'";
        expect_parse_error(sql, "sql parser error: Expected ',' or ')' after partition definition, found: int");

        // positive case: additional options (one entry) can be specified
        let sql =
            "CREATE EXTERNAL TABLE t STORED AS x OPTIONS ('k1' 'v1') LOCATION 'blahblah'";
        let expected = Statement::CreateExternalTable(CreateExternalTable {
            name: "t".into(),
            columns: vec![],
            file_type: "X".to_string(),
            has_header: false,
            delimiter: ',',
            location: "blahblah".into(),
            table_partition_cols: vec![],
            if_not_exists: false,
            file_compression_type: "".to_string(),
            options: HashMap::from([("k1".into(), "v1".into())]),
        });
        expect_parse_ok(sql, expected)?;

        // positive case: additional options (multiple entries) can be specified
        let sql =
            "CREATE EXTERNAL TABLE t STORED AS x OPTIONS ('k1' 'v1', k2 v2) LOCATION 'blahblah'";
        let expected = Statement::CreateExternalTable(CreateExternalTable {
            name: "t".into(),
            columns: vec![],
            file_type: "X".to_string(),
            has_header: false,
            delimiter: ',',
            location: "blahblah".into(),
            table_partition_cols: vec![],
            if_not_exists: false,
            file_compression_type: "".to_string(),
            options: HashMap::from([
                ("k1".into(), "v1".into()),
                ("k2".into(), "v2".into()),
            ]),
        });
        expect_parse_ok(sql, expected)?;

        // Error cases: partition column does not support type
        let sql =
            "CREATE EXTERNAL TABLE t STORED AS x OPTIONS ('k1' 'v1', k2 v2, k3) LOCATION 'blahblah'";
        expect_parse_error(sql, "sql parser error: Expected literal string, found: )");

        // Error case: `with header` is an invalid syntax
        let sql = "CREATE EXTERNAL TABLE t STORED AS CSV WITH HEADER LOCATION 'abc'";
        expect_parse_error(sql, "sql parser error: Expected LOCATION, found: WITH");

        // Error case: `partitioned` is an invalid syntax
        let sql = "CREATE EXTERNAL TABLE t STORED AS CSV PARTITIONED LOCATION 'abc'";
        expect_parse_error(
            sql,
            "sql parser error: Expected LOCATION, found: PARTITIONED",
        );

        Ok(())
    }
}
