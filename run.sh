cargo test --release ramlookup::cq::tests::test_setup -- --show-output --nocapture > test_setup.txt

cargo test --release --package caulk --lib ramlookup::cq::tests::test_cq_table_params -- --exact --nocapture > test_cq_table_params.txt