mod classification_tree;
mod regression_tree;

use smartcore::linalg::basic::matrix::DenseMatrix;
use regression_tree::{RDataset, train_regression_tree, reg_read_csv};
use classification_tree::{CDataset, class_read_csv};
use std::error::Error;
use smartcore::metrics::accuracy;
use smartcore::model_selection::train_test_split;
use smartcore::tree::decision_tree_classifier::DecisionTreeClassifier;

fn classification_workflow(file_path: &str, target_column_index: usize) -> Result<(), Box<dyn Error>> {
    // 1. Read Classification Dataset
    let (mut class_train_dataset, mut class_prediction_dataset) = 
        class_read_csv(file_path, target_column_index)?;

    // 2. Scale Features (Skipping specific columns)
    let classification_categorical_columns = vec![0, 2, 5];
    class_train_dataset.c_scale_features(&classification_categorical_columns);
    class_prediction_dataset.c_scale_features(&classification_categorical_columns);

    // 3. Subset Features
    let classification_feature_columns = vec![1, 3, 7, 9, 12, 21, 22, 33, 34];
    let class_subset = class_train_dataset.c_subset(&classification_feature_columns);

    // 4. Train Decision Tree Classifier
    let features = class_subset.to_dense_matrix();
    let target = class_subset.to_target_vector();

    let (x_train, x_test, y_train, y_test) = train_test_split(&features, &target, 0.8, true, None);

    let classification_model = DecisionTreeClassifier::fit(&x_train, &y_train, Default::default());
    if let Ok(classification_model) = classification_model {
        println!("Classification training successful!");
        // Evaluate model
        let predictions = classification_model.predict(&x_test)?;
        let accuracy_score = accuracy(&y_test, &predictions);
        println!("Accuracy: {:.2}%", accuracy_score * 100.0);

        // Predict Future Games
        let prediction_subset = class_prediction_dataset.c_subset(&classification_feature_columns);
        let prediction_features = DenseMatrix::from_2d_vec(&prediction_subset.features);
        let future_predictions = classification_model.predict(&prediction_features)?;
        println!("Predictions for future games: {:?}", future_predictions);
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let file_path = "final w: preds.csv";
    
    for i in [4, 8, 16].iter() {
    // Regression workflow
    let reg_target_column_index = *i;
    let (mut train_dataset, mut prediction_dataset) = reg_read_csv(file_path, reg_target_column_index)?;

    train_dataset.r_scale_features();
    prediction_dataset.r_scale_features();
    let regression_feature_columns = match reg_target_column_index {
        4 => vec![0, 1, 2, 5, 6, 20, 25, 26, 37, 38],
        8 => vec![0, 1, 3, 6, 7, 9, 10, 13, 14, 17, 21, 22, 25, 26, 29, 30, 33,34],
        16 => vec![0, 1, 3, 6,7, 17, 18, 20,21,22,25,26,37,38,40],         // Feature set for target column 16 (different set)
         _ => vec![0, 1, 2, 5, 6, 20, 25, 26, 37, 38], // Default feature set (if needed)
        };
    let reg_subset = train_dataset.r_subset(&regression_feature_columns);

    let regression_model = train_regression_tree(&reg_subset)?;
    let reg_prediction_subset = prediction_dataset.r_subset(&regression_feature_columns);
    let reg_predictions = regression_model.predict(&DenseMatrix::from_2d_vec(&reg_prediction_subset.features))?;

    println!("Regression predictions for column index {:?} for future games: {:?}", i, reg_predictions);
        
    }

    // Classification workflow (Result)
    let class_target_column_index = 2;
    classification_workflow(file_path, class_target_column_index)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
#[test]
fn test_c_subset() {
    let features = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ];
    let target = vec![0, 1, 2];
    let dataset = CDataset::new(features.clone(), target.clone());
    
    let subset = dataset.c_subset(&[0, 2]);
    
    assert_eq!(
        subset.features,
        vec![
            vec![1.0, 3.0],
            vec![4.0, 6.0],
            vec![7.0, 9.0],
        ]
    );
    assert_eq!(subset.target, target);
}

    #[test]
fn test_c_scale_features() {
    let mut features = vec![
        vec![10.0, 1.0, 3.0],
        vec![20.0, 2.0, 6.0],
        vec![30.0, 3.0, 9.0],
    ];
    let target = vec![0, 1, 2];
    let mut dataset = CDataset::new(features.clone(), target);

    dataset.c_scale_features(&[1]); // Column 1 is categorical

    let expected_features = vec![
        vec![0.0, 1.0, 0.0],
        vec![0.5, 2.0, 0.5],
        vec![1.0, 3.0, 1.0],
    ];

    assert_eq!(dataset.features, expected_features);
}

    #[test]
fn test_train_regression_tree() {
    let features = vec![
        vec![1.0, 2.0],
        vec![3.0, 4.0],
        vec![5.0, 6.0],
        vec![7.0, 8.0],
        vec![9.0, 10.0],
    ];
    let target = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let dataset = RDataset::new(features, target);

    let model = train_regression_tree(&dataset).expect("Failed to train regression tree");
    
    let test_data = DenseMatrix::from_2d_vec(&vec![vec![2.0, 3.0], vec![6.0, 7.0]]);
    let predictions = model.predict(&test_data).expect("Failed to make predictions");

    assert!(predictions.len() == 2); // Ensure predictions are made for each row
    println!("Predictions: {:?}", predictions);
}
}