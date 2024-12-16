use std::collections::{HashMap};
use smartcore::linalg::basic::matrix::DenseMatrix;
use csv::ReaderBuilder;
use std::error::Error;
use chrono::NaiveDate;

#[derive(Debug)]
pub struct CDataset {
    pub features: Vec<Vec<f64>>, 
    pub target: Vec<i32>,       
}

impl CDataset {
    pub fn new(features: Vec<Vec<f64>>, target: Vec<i32>) -> Self {
        CDataset { features, target }
    }

    pub fn c_subset(&self, columns: &[usize]) -> Self {
        let subset_features = self.features.iter()
            .map(|row| columns.iter().map(|&col| row[col]).collect::<Vec<f64>>())
            .collect::<Vec<Vec<f64>>>();
        let subset_target = self.target.clone();
        CDataset::new(subset_features, subset_target)
    }

    pub fn to_dense_matrix(&self) -> DenseMatrix<f64> { 
        DenseMatrix::from_2d_vec(&self.features)
    }

    pub fn to_target_vector(&self) -> Vec<i32> {
        self.target.clone()
    }
    
    pub fn c_scale_features(&mut self, categorical_columns: &[usize]) {
    for col in 0..self.features[0].len() {
        if !categorical_columns.contains(&col) { // Skipped categorical columns
            let min_val = self.features.iter().map(|row| row[col] as f64).fold(f64::INFINITY, f64::min);
            let max_val = self.features.iter().map(|row| row[col] as f64).fold(f64::NEG_INFINITY, f64::max);
            
            for row in self.features.iter_mut() {
                row[col] = (row[col] as f64 - min_val) / (max_val - min_val);
            }
        }
    }
}

}

pub fn class_read_csv(file_path: &str, target_column_index: usize) -> Result<(CDataset, CDataset), Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new().has_headers(true).from_path(file_path)?;

    let mut features = Vec::new();
    let mut target = Vec::new();
    let mut prediction_features = Vec::new();

    let league_averages: HashMap<&str, f64> = HashMap::from([
        ("PP%", 20.69),
        ("PK%", 78.96),
        ("Net PP%", 17.71),
        ("Net PK%", 81.67),
        ("T2PP%", 20.69),
        ("T2PK%", 78.96),
        ("T2Net PP%", 17.71),
        ("T2Net PK%", 81.67),
    ]);

    let headers = rdr.headers()?.clone();

    // Categorical column encoding
    let mut team_name_encoding: HashMap<String, i32> = HashMap::new();
    let mut team_name_counter = 0;

    for (index, result) in rdr.records().enumerate() {
        let record = result?;

        let row: Result<Vec<f64>, _> = record.iter()
            .enumerate()
            .filter(|(i, _)| *i != target_column_index) // Skip the target column
            .map(|(col_index, field)| {
                let field_str = field.trim().to_string();

                if field_str == "--" {
                    if let Some(column_name) = headers.get(col_index) {
                        if let Some(&average) = league_averages.get(column_name) {
                            return Ok(average);
                        }
                    }
                    Ok(0.0)
                } else if let Ok(parsed_date) = NaiveDate::parse_from_str(&field_str, "%m/%d/%Y") {
                   
                    let parsed_datetime = parsed_date.and_hms_opt(0, 0, 0).unwrap_or_else(|| {
                        NaiveDate::from_ymd_opt(1970, 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap()
                    });
                    Ok(parsed_datetime.and_utc().timestamp() as f64)
                } else if field_str.parse::<f64>().is_ok() {
                   
                    field_str.parse::<f64>()
                } else {
                    if let Some(existing_code) = team_name_encoding.get(&field_str) {
                        Ok(*existing_code as f64)
                    } else {
                        team_name_counter += 1;
                        team_name_encoding.insert(field_str.clone(), team_name_counter);
                        Ok(team_name_counter as f64)
                    }
                }
            })
            .collect();

        if let Ok(row) = row {
            if index < 6 {
                // First two rows are prediction rows
                prediction_features.push(row);
            } else {
                let target_value = record.get(target_column_index).and_then(|field| {
                    field.parse::<i32>().ok() // Parse target as i32 for classification
                });

                if let Some(target_value) = target_value {
                    features.push(row);
                    target.push(target_value);
                } //else {
                    //eprintln!("Skipping invalid row {}: {:?}", index, record);
                //}
            }
        } else {
            eprintln!("Error parsing row {}: {:?}", index, record);
        }
    }

    let dataset = CDataset::new(features, target);
    let prediction_dataset = CDataset::new(prediction_features, Vec::new()); // No target for predictions

    Ok((dataset, prediction_dataset))
}

