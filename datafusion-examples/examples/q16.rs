

use datafusion::error::Result;
use datafusion::prelude::*;

/// This example demonstrates q16 bug, i.e mismatch between names, types and nullability
/// 
/// To expose the bug add the folowing println! to datafusion/core/src/datasource/memory.rs to see an nice outpout of the differnet behavior...

// // impl MemTable {
//     /// Create a new in-memory table from the provided schema and record batches
//     pub fn try_new(schema: SchemaRef, partitions: Vec<Vec<RecordBatch>>) -> Result<Self> {
//         // println!(
//         //     "\nTRY NEW MEMTABLE\n  LP    {:?}\n  BATCH {:?} \n",
//         //     schema,
//         //     if partitions.len() > 0 && partitions[0].len() > 0 {
//         //         partitions[0][0].schema()
//         //     } else {
//         //         Arc::new(Schema::new(vec![]))
//         //     }
//         // );

#[tokio::main]
async fn main() -> Result<()> {
    // create local execution context
    let session_config = SessionConfig::from_env()?
        .with_create_default_catalog_and_schema(true)
        .with_information_schema(true);
    let ctx: SessionContext = SessionContext::with_config(session_config);

    let sqls = vec![
        ("a", "CREATE TABLE a AS VALUES (cast(1 as smallint)), (cast(0 as smallint)), (cast(-10 as smallint));"),
        
        ("s1", "CREATE TABLE s1 AS SELECT signum(column1) FROM a;"),
        ("", "SELECT * FROM s1;"),
        ("s2", "CREATE TABLE s2 AS SELECT signum(column1) as xx FROM a;"),
        ("", "SELECT * FROM s2;"),
        
        ("c1", "CREATE TABLE c1 AS SELECT count(column1) FROM a;"),
        ("", "SELECT * FROM c1;"), 
        ("c2", "CREATE TABLE c2 AS SELECT count(column1) as xx FROM a;"),
        ("", "SELECT * FROM c2;"), 

        ("a1", "CREATE TABLE a1 AS SELECT approx_distinct(column1) FROM a;"),
        ("", "SELECT * FROM a1;"), 
        ("a2", "CREATE TABLE a2 AS SELECT approx_distinct(column1) AS xx FROM a;"),
        ("", "SELECT * FROM a2;"),


    ];

    for (_id, sql) in sqls {
        // execute the query
        println!("\nINPUT SQL: {:?}", sql);
        let df = ctx.sql(sql).await?;
        df.show().await?;
    }

    Ok(())
}
