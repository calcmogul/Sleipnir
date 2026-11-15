// Copyright (c) Sleipnir contributors

// Its own test binary, since spy writes files to the working directory

use std::fs;

use sleipnir::{Options, Problem, SolveStatus, constrain};

#[test]
fn spy_writes_sparsity_patterns() {
    let files = ["H.spy", "A_e.spy", "A_i.spy"];
    for file in files {
        let _ = fs::remove_file(file);
    }

    let problem: Problem = Problem::new();
    let x = problem.decision_variable_matrix(2, 1);
    problem.minimize(x.dot(&x));
    problem.subject_to(constrain!(x[0] + x[1] == 1.0));
    problem.subject_to(constrain!(x >= 0.0));
    let options = Options {
        spy: true,
        ..Default::default()
    };
    assert_eq!(problem.solve(&options), Ok(SolveStatus::Success));

    for file in files {
        let metadata = fs::metadata(file).unwrap_or_else(|e| panic!("{file}: {e}"));
        assert!(metadata.len() > 0, "{file} is empty");
        fs::remove_file(file).unwrap();
    }
}
