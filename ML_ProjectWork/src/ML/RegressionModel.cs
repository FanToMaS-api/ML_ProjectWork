using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using ML_ProjectWork.Helpers;
using ML_ProjectWork.ML.Enum;
using ML_ProjectWork.Models;

namespace ML_ProjectWork.ML
{
    /// <summary>
    ///     Класс занимается созданием и обучением модели, предсказанием результатов
    /// </summary>
    public class RegressionModel : IModel
    {
        #region Fields

        /// <summary>
        ///     Отношение данных: тестовые к тренировочным
        /// </summary>
        private readonly double _testFraction;

        /// <summary>
        ///     Кол-во фичей, которые мы не будет учитывать при обучении
        /// </summary>
        private readonly int _countExcludedFeatures;

        /// <summary>
        ///     Список исключенных фичей
        /// </summary>
        private readonly List<string> _dropColumns;

        /// <summary>
        ///     Данные
        /// </summary>
        private readonly IDataView _dataView;

        /// <summary>
        ///     Разделитель данных в файле
        /// </summary>
        private readonly char _separatorChar;

        /// <summary>
        ///     Словарь возвращающий тренеров по типу
        /// </summary>
        private static readonly Dictionary<TrainerModel, IEstimator<ITransformer>> _trainers =
            new()
            {
                { TrainerModel.LbfgsPoissonRegression, new MLContext(212103).Regression.Trainers.LbfgsPoissonRegression() },
                { TrainerModel.FastForest, new MLContext(212103).Regression.Trainers.FastForest() },
                { TrainerModel.FastTree, new MLContext(212103).Regression.Trainers.FastTree() },
                { TrainerModel.FastTreeTweedie, new MLContext(212103).Regression.Trainers.FastTreeTweedie() },
                { TrainerModel.Gam, new MLContext(212103).Regression.Trainers.Gam() },
                { TrainerModel.LightGbm, new MLContext(212103).Regression.Trainers.LightGbm() },
                { TrainerModel.Ols, new MLContext(212103).Regression.Trainers.Ols() },
                { TrainerModel.OnlineGradientDescent, new MLContext(212103).Regression.Trainers.OnlineGradientDescent() },
                { TrainerModel.Sdca, new MLContext(212103).Regression.Trainers.Sdca() }
            };

        /// <summary>
        ///     Словарь возвращающий пути сохранения моделей по типу трененров
        /// </summary>
        private static readonly Dictionary<TrainerModel, string> _trainerPath =
            new()
            {
                { TrainerModel.LbfgsPoissonRegression, "LbfgsPoissonRegression.zip" },
                { TrainerModel.FastForest, "FastForest.zip" },
                { TrainerModel.FastTree, "FastTree.zip" },
                { TrainerModel.FastTreeTweedie, "FastTreeTweedie.zip" },
                { TrainerModel.Gam, "Gam.zip" },
                { TrainerModel.LightGbm, "LightGbm.zip" },
                { TrainerModel.Ols, "Ols.zip" },
                { TrainerModel.OnlineGradientDescent, "OnlineGradientDescent.zip" },
                { TrainerModel.Sdca, "Sdca.zip" }
            };

        #endregion

        #region .ctor

        /// <inheritdoc cref="RegressionModel"/>
        public RegressionModel(
            string dataPath,
            int countExcludedFeatures,
            TrainerModel trainer,
            bool isLoadSavedModel,
            List<int> anomalyIndexes,
            char separatorChar = ',',
            double testFraction = 0.2)
        {
            MlContext = new MLContext(212103); // 212103 - seed, чтобы при новом запуске результаты оставались теми же
            IsLoadSavedModel = isLoadSavedModel;
            Trainer = trainer;

            _testFraction = testFraction;
            _dropColumns = new();
            _countExcludedFeatures = countExcludedFeatures;
            _separatorChar = separatorChar;

            var housesData = LoadData(dataPath, anomalyIndexes);
            _dataView = MlContext.Data.LoadFromEnumerable(housesData);

            // Изначально отбираю все фичи
            Features = _dataView.Schema.Select(_ => _.Name).ToList();
        }

        #endregion

        #region Properties

        /// <inheritdoc />
        public double MeanAbsoluteError { get; private set; }

        /// <inheritdoc />
        public double RSquared { get; private set; }

        /// <inheritdoc />
        public double RootMeanSquaredError { get; private set; }

        /// <inheritdoc />
        public List<string> Features { get; }

        /// <inheritdoc />
        public ITransformer Model { get; private set; }

        /// <inheritdoc />
        public MLContext MlContext { get; }

        /// <inheritdoc />
        public TrainerModel Trainer { get; }

        /// <inheritdoc />
        public IDataView PredictedData { get; private set; }

        /// <inheritdoc />
        public IDataView TrainData { get; private set; }

        /// <inheritdoc />
        public IDataView TestData { get; private set; }

        /// <inheritdoc />
        public bool IsLoadSavedModel { get; init; }

        #endregion

        #region Public methods

        /// <inheritdoc />
        public void Fit()
        {
            // Разделение данных на тестовые и тренировочные в соответствии с фракцией
            var trainTestData = MlContext.Data.TrainTestSplit(_dataView, _testFraction);
            TrainData = trainTestData.TrainSet;
            TestData = trainTestData.TestSet;

            if (IsLoadSavedModel)
            {
                if (!File.Exists(_trainerPath[Trainer]))
                {
                    throw new FileNotFoundException("Error! Нет данных для загрузке модели для данного тренера.");
                }

                Load();
            }
            else
            {
                DataPreparing();
                var trainedPipeline = CreatePipeline();

                // Обучение модели
                Model = trainedPipeline.Fit(TrainData);
            }

            // Тестирование точности на тестовых данных
            PredictedData = Model.Transform(TestData);

            var metrics = MlContext.Regression.Evaluate(PredictedData);

            MeanAbsoluteError = metrics.MeanAbsoluteError;
            RSquared = metrics.RSquared;
            RootMeanSquaredError = metrics.RootMeanSquaredError;
        }

        /// <inheritdoc />
        public IEstimator<ITransformer> SetTrainer()
        {
            if (!_trainers.TryGetValue(Trainer, out var trainer))
            {
                throw new Exception("Unknown trainer");
            }
            else
            {
                return trainer;
            }
        }

        /// <inheritdoc />
        public void Save()
        {
            MlContext.Model.Save(Model, _dataView.Schema, _trainerPath[Trainer]);
        }

        #endregion

        #region Private methods

        /// <summary>
        ///     Загружает данные и чистит их от аномалий
        /// </summary>
        private HouseModel[] LoadData(string dataPath, List<int> anomalyIndexes)
        {
            // Получаю и обрабатываю данные
            var housesData = File.ReadAllLines(dataPath)
                .Skip(1)
                .Select(ProcessingData)
                .ToList();

            if (anomalyIndexes is null)
            {
                return housesData.ToArray();
            }

            for (var i = 0; i < anomalyIndexes.Count; i++)
            {
                var anomalyIndex = anomalyIndexes[i];
                housesData.RemoveAt(anomalyIndex - i); // так как после первого удаления все смещается
            }

            return housesData.ToArray();
        }

        /// <summary>
        ///     Приводит выбивающиеся фичи к верному типу
        /// </summary>
        public HouseModel ProcessingData(string houseInfo)
        {
            var information = houseInfo.Split(_separatorChar);

            return new HouseModel
            {
                Id = float.Parse(information[0].Replace("\"", "")),
                Price = float.Parse(information[2]),
                Bedrooms = float.Parse(information[3]),
                Bathrooms = float.Parse(information[4]),
                LivingArea = float.Parse(information[5]),
                Area = float.Parse(information[6]),
                Floors = float.Parse(information[7].Replace("\"", "")),
                IsWaterFront = float.Parse(information[8]),
                View = float.Parse(information[9]),
                Condition = float.Parse(information[10]),
                Grade = float.Parse(information[11]),
                SqftAbove = float.Parse(information[12]),
                SqftBasement = float.Parse(information[13]),
                YearBuilt = float.Parse(information[14]),
                YearRenovation = float.Parse(information[15].Replace("\"", "")),
                ZipCode = float.Parse(information[16].Replace("\"", "")),
                Lat = float.Parse(information[17]),
                Long = float.Parse(information[18]),
                SqftLiving15 = float.Parse(information[19]),
                SqftLot15 = float.Parse(information[20]),
            };
        }

        /// <summary>
        ///     Готовит данные к обучению
        /// </summary>
        private void DataPreparing()
        {
            //  Получение названий фич в порядке уменьшения влияния на Label
            var sortedFeatures = FeatureHelper.FindBestFeatures(this, _dataView);
            if (_countExcludedFeatures >= sortedFeatures.Count)
            {
                return;
            }

            //  Добавление фич для сброса, которые учитываться не будут
            for (var i = 0; i < _countExcludedFeatures; i++)
            {
                _dropColumns.Add(sortedFeatures[sortedFeatures.Count - 1 - i]);
            }

            Features.Clear();
            for (var i = 0; i < sortedFeatures.Count - _countExcludedFeatures; i++)
            {
                Features.Add(sortedFeatures[i]);
            }

            // Нельзя включать Label в Features
            if (Features.Contains("Label"))
            {
                Features.Remove("Label");
            }
        }

        /// <summary>
        ///     Создает pipeline
        /// </summary>
        private EstimatorChain<ITransformer> CreatePipeline()
        {
            // Подготовка основного pipline
            EstimatorChain<NormalizingTransformer> pipeline;
            if (_dropColumns.Count == 0)
            {
                pipeline = MlContext.Transforms.Concatenate("Features", Features.ToArray())

                    // Выбор нормализации влияет на результат
                    // https://docs.microsoft.com/en-us/dotnet/api/microsoft.ml.transforms.normalizingestimator?view=ml-dotnet
                    .Append(MlContext.Transforms.NormalizeBinning("Features"));
            }
            else
            {
                pipeline = MlContext.Transforms.Concatenate("Features", Features.ToArray())
                    .Append(MlContext.Transforms.DropColumns(_dropColumns.ToArray()))
                     .Append(MlContext.Transforms.NormalizeBinning("Features"));
            }

            var trainer = SetTrainer();
            return pipeline.Append(trainer);
        }

        /// <summary>
        ///     Загружает модель из файла
        /// </summary>
        private void Load()
        {
            Model = MlContext.Model.Load(_trainerPath[Trainer], out _);
        }

        #endregion
    }
}
