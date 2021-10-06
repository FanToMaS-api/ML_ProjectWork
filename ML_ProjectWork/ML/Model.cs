using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.LightGbm;
using System;
using System.Collections.Generic;
using System.Linq;

namespace ML_ProjectWork
{
    /// <summary>
    ///     Занимается созданием и обучением модели
    /// </summary>
    internal class Model
    {
        #region Fields

        private readonly string _dataPath;

        private readonly TrainerModel _trainer;

        private readonly double _testFraction;

        private readonly char _separatorChar;

        private readonly int _countExcludedFeatures;

        private string[] _featureNames;

        private IDataView? _dataView;

        private readonly MLContext _mlContext;

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
        public Model(string dataPath, int countExcludedFeatures, TrainerModel trainer, char separatorChar = ',', double testFraction = 0.2)
        {
            _mlContext = new MLContext(212103);
            _dataPath = dataPath;
            _countExcludedFeatures = countExcludedFeatures;
            _trainer = trainer;

            _separatorChar = separatorChar;
            _testFraction = testFraction;
            _dataView = _mlContext.Data.LoadFromTextFile<HouseModel>(_dataPath, _separatorChar);
            _featureNames = _dataView.Schema.Select(_ => _.Name).ToArray();
        }

        #endregion

        #region Properties

        /// <summary>
        ///     Точность модели
        /// </summary>
        public float Accuracy { get; set; }

        #endregion

        #region Public methods

        /// <summary>
        ///     Проводит обучение модели
        /// </summary>
        public void Fit()
        {
            var excludedFeatures = FindBestFeatures();
            if (_countExcludedFeatures > excludedFeatures.Count)
            {
                
            }
        }

        #endregion

        #region Private methods

        /// <summary>
        ///     Тренирует модель, находит индексы самых значимых фичей
        /// </summary>
        private List<string> FindBestFeatures()
        {
            var result = new List<string>();
            if (_trainer == TrainerModel.LbfgsPoissonRegression)
            {
                return result;
            }

            var pipeline = _mlContext.Transforms.Concatenate("Features", _featureNames)
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
        ///     Готовит данные к обучению
        /// </summary>
        private void DataPreparing()
        {

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

        #endregion
    }
}
