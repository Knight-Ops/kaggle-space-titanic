use crate::{
    data::TitanicBatcher,
    dataset::{TitanicDataset, TitanicItem},
    model::Model,
    training::TrainingConfig,
};
use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    prelude::*,
    record::{CompactRecorder, NoStdInferenceRecorder, NoStdTrainingRecorder, Recorder},
};

pub fn infer<B: Backend<IntElem = i64>>(
    artifact_dir: &str,
    device: B::Device,
    dataset: TitanicDataset,
) {
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist");

    let model: Model<B> = config.model.init(&device).load_record(record);

    let batcher = TitanicBatcher::new(device);
    println!("PassengerId,Transported");
    for item in dataset.iter() {
        let batch = batcher.batch(vec![item.clone()]);
        let output = model.forward(batch.inputs);
        let predicted = output.argmax(1).flatten::<1>(0, 1).into_scalar();

        let output = format!(
            "{:04}_{:02},{}",
            &item.group_number,
            &item.passenger_number,
            match predicted {
                1 => "True",
                _ => "False",
            }
        );

        println!("{}", output);
    }
}
