using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.LightGbm;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace ML_ProjectWork
{
    /// <summary>
    ///     Занимается созданием и обучением модели
    /// </summary>
    internal class Model
    {
        #region Fields

        private readonly TrainerModel _trainer;

        private readonly double _testFraction;

        private readonly int _countExcludedFeatures;

        private readonly List<string> _featureNames;

        private readonly List<string> _dropColumns;

        private readonly IDataView? _dataView;

        private readonly MLContext _mlContext;

        private readonly string _label;

        private readonly char _separatorChar;

        private static Dictionary<TrainerModel, IEstimator<ITransformer>> _trainers =
            new()
            {
                { TrainerModel.LbfgsPoissonRegression, new MLContext(212103).Regression.Trainers.LbfgsPoissonRegression() },
                { TrainerModel.FastForest, new MLContext(212103).Regression.Trainers.LightGbm() },
                { TrainerModel.FastTree, new MLContext(212103).Regression.Trainers.FastTree() },
                { TrainerModel.FastTreeTweedie, new MLContext(212103).Regression.Trainers.FastTreeTweedie() },
                { TrainerModel.Gam, new MLContext(212103).Regression.Trainers.Gam() },
                { TrainerModel.LightGbm, new MLContext(212103).Regression.Trainers.LightGbm() },
                { TrainerModel.Ols, new MLContext(212103).Regression.Trainers.Ols() },
                { TrainerModel.OnlineGradientDescent, new MLContext(212103).Regression.Trainers.OnlineGradientDescent() },
                { TrainerModel.Sdca, new MLContext(212103).Regression.Trainers.Sdca() }
            };

        #endregion

        #region .ctor

        /// <inheritdoc cref="Model"/>
        public Model(string dataPath, int countExcludedFeatures, TrainerModel trainer, string label, char separatorChar = ',', double testFraction = 0.2)
        {
            _mlContext = new MLContext(212103);
            _countExcludedFeatures = countExcludedFeatures;
            _trainer = trainer;

            _label = label;
            _testFraction = testFraction;
            _dropColumns = new();
            _separatorChar = separatorChar;

            var housesData = File.ReadAllLines(dataPath)
                .Skip(1)
                .Select(ProcessingData)
                .ToArray();

            _dataView = _mlContext.Data.LoadFromEnumerable(housesData);

            //_dataView = _mlContext.Data.LoadFromTextFile<HouseModel>(dataPath, separatorChar);
            _featureNames = _dataView.Schema.Select(_ => _.Name).ToList();
        }

        #endregion

        #region Properties

        /// <summary>
        ///     Точность модели
        /// </summary>
        public double Accuracy { get; set; }

        #endregion

        #region Public methods

        /// <summary>
        ///     Проводит обучение модели
        /// </summary>
        public void Fit()
        {
            DataPreparing();
            var trainedPipeline = CreatePipeline();
            var trainTestData = _mlContext.Data.TrainTestSplit(_dataView, _testFraction);
            var trainData = trainTestData.TrainSet;
            var testsData = trainTestData.TestSet;
            var model = trainedPipeline.Fit(trainData);

            // Тестирование точности
            var predsDataView = model.Transform(testsData);
            var metrics = _mlContext.Regression.Evaluate(predsDataView);

            Accuracy = metrics.RSquared;
        }

        #endregion

        #region Private methods

        /// <summary>
        ///     Приводит выбивающиеся фичи к верному типу
        /// </summary>
        private HouseModel ProcessingData(string houseInfo)
        {
            var information = houseInfo.Split(_separatorChar);

        }

        /// <summary>
        ///     Готовит данные к обучению
        /// </summary>
        private void DataPreparing()
        {
            //  Получение названий фич в порядке уменьшения влияния на Label
            var sortedFeatures = FindBestFeatures();
            if (_countExcludedFeatures >= sortedFeatures.Count)
            {
                return;
            }

            //  Добавление фич для сброса, которые учитываться не будут
            for (var i = _countExcludedFeatures; i < _featureNames.Count; i++)
            {
                _dropColumns.Add(sortedFeatures[i]);
            }

            _featureNames.Clear();
            for (var i = 0; i < _countExcludedFeatures; i++)
            {
                _featureNames.Add(sortedFeatures[i]);
            }

            if (!_featureNames.Contains(_label))
            {
                _featureNames.Add(_label);
            }
        }

        /// <summary>
        ///     Тренирует модель, находит индексы самых значимых фичей
        /// </summary>
        private List<string> FindBestFeatures()
        {
            var result = new List<string>();

            var pipeline = _mlContext.Transforms.Concatenate("Features", _featureNames.ToArray())
                .Append(_mlContext.Transforms.NormalizeLogMeanVariance("Features"));

            var trainer = SetTrainer();
            var trainedPipeline = pipeline.Append(trainer);

            var model = trainedPipeline.Fit(_dataView);
            var preprocessedData = model.Transform(_dataView);
            var indexes = GetIndexes(trainer, preprocessedData);

            foreach (var index in indexes)
            {
                result.Add(_featureNames[index]);
            }

            return result;
        }

        /// <summary>
        ///     Возвращает индексы фич в порядке убывания значимости
        /// </summary>
        private List<int> GetIndexes(IEstimator<ITransformer> trainer, IDataView preprocessedData)
        {
            if (_trainer == TrainerModel.FastForest)
            {
                return FeatureHelper.FeaturePermutation(
                    _mlContext,
                    (RegressionPredictionTransformer<FastForestRegressionModelParameters>)trainer.Fit(preprocessedData),
                    preprocessedData);
            }

            if (_trainer == TrainerModel.LbfgsPoissonRegression)
            {
                return FeatureHelper.FeaturePermutation(
                    _mlContext,
                    (RegressionPredictionTransformer<PoissonRegressionModelParameters>)trainer.Fit(preprocessedData),
                    preprocessedData);
            }

            if (_trainer == TrainerModel.Sdca)
            {
                return FeatureHelper.FeaturePermutation(
                    _mlContext,
                    (RegressionPredictionTransformer<LinearRegressionModelParameters>)trainer.Fit(preprocessedData),
                    preprocessedData);
            }

            if (_trainer == TrainerModel.FastTree)
            {
                return FeatureHelper.FeaturePermutation(
                    _mlContext,
                    (RegressionPredictionTransformer<FastTreeRegressionModelParameters>)trainer.Fit(preprocessedData),
                    preprocessedData);
            }

            if (_trainer == TrainerModel.FastTreeTweedie)
            {
                return FeatureHelper.FeaturePermutation(
                    _mlContext,
                    (RegressionPredictionTransformer<FastTreeTweedieModelParameters>)trainer.Fit(preprocessedData),
                    preprocessedData);
            }

            if (_trainer == TrainerModel.Gam)
            {
                return FeatureHelper.FeaturePermutation(
                    _mlContext,
                    (RegressionPredictionTransformer<GamRegressionModelParameters>)trainer.Fit(preprocessedData),
                    preprocessedData);
            }

            if (_trainer == TrainerModel.Ols)
            {
                return FeatureHelper.FeaturePermutation(
                    _mlContext,
                    (RegressionPredictionTransformer<OlsModelParameters>)trainer.Fit(preprocessedData),
                    preprocessedData);
            }

            if (_trainer == TrainerModel.OnlineGradientDescent)
            {
                return FeatureHelper.FeaturePermutation(
                    _mlContext,
                    (RegressionPredictionTransformer<LinearRegressionModelParameters>)trainer.Fit(preprocessedData),
                    preprocessedData);
            }

            if (_trainer == TrainerModel.LightGbm)
            {
                return FeatureHelper.FeaturePermutation(
                    _mlContext,
                    (RegressionPredictionTransformer<LightGbmRegressionModelParameters>)trainer.Fit(preprocessedData),
                    preprocessedData);
            }

            return new List<int>();
        }

        /// <summary>
        ///     Устанавливает тренера в соответствии с
        /// </summary>
        private IEstimator<ITransformer> SetTrainer()
        {
            if (!_trainers.TryGetValue(_trainer, out var trainer))
            {
                throw new Exception("Unknown trainer");
            }
            else
            {
                return trainer;
            }
        }

        private EstimatorChain<ITransformer> CreatePipeline()
        {
            var pipeline = _mlContext.Transforms.Concatenate("Features", _featureNames.ToArray())
                .Append(_mlContext.Transforms.DropColumns(_dropColumns.ToArray()))
                .Append(_mlContext.Transforms.NormalizeLogMeanVariance("Features"));

            var trainer = SetTrainer();
            return pipeline.Append(trainer);
        }

        #endregion
    }
}
