using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.LightGbm;
using System;
using System.Collections.Generic;
using System.Linq;

namespace ML_ProjectWork.Helpers
{
    /// <summary>
    ///     Вычисляет значимость фич
    /// </summary>
    internal static class FeatureHelper
    {
        /// <summary>
        ///     Выводит фичи в порядке уменьшения влияния на результат
        /// </summary>
        /// <param name="permutationCount"> Кол-во фич для расчета </param>
        public static List<int> FeaturePermutation(
            MLContext mlColntext,
            RegressionPredictionTransformer<LinearModelParameters> model,
            IDataView preprocessedTrainData,
            int permutationCount)
        {
            var featuresPermutation = mlColntext.Regression.PermutationFeatureImportance(
                model,
                preprocessedTrainData,
                permutationCount: permutationCount);

            return featuresPermutation
                .Select((metric, index) => new { index, metric.RSquared })
                .OrderByDescending(_ => Math.Abs(_.RSquared.Mean))
                .Select(_ => _.index).ToList();
        }

        /// <summary>
        ///     Sdca & OnlineGradientDescent: Выводит фичи в порядке уменьшения влияния на результат
        /// </summary>
        /// <param name="permutationCount"> Кол-во фич для расчета </param>
        public static List<int> FeaturePermutation(
            MLContext mlColntext,
            RegressionPredictionTransformer<LinearRegressionModelParameters> model,
            IDataView preprocessedTrainData,
            int permutationCount)
        {
            var featuresPermutation = mlColntext.Regression.PermutationFeatureImportance(
                model,
                preprocessedTrainData,
                permutationCount: permutationCount);

            return featuresPermutation
                .Select((metric, index) => new { index, metric.RSquared })
                .OrderByDescending(_ => Math.Abs(_.RSquared.Mean))
                .Select(_ => _.index).ToList();
        }

        /// <summary>
        ///     LightGbm: Выводит фичи в порядке уменьшения влияния на результат
        /// </summary>
        /// <param name="permutationCount"> Кол-во фич для расчета </param>
        public static List<int> FeaturePermutation(
            MLContext mlColntext,
            RegressionPredictionTransformer<LightGbmRegressionModelParameters> model,
            IDataView preprocessedTrainData,
            int permutationCount)
        {
            var featuresPermutation = mlColntext.Regression.PermutationFeatureImportance(
                model,
                preprocessedTrainData,
                permutationCount: permutationCount);

            return featuresPermutation
                .Select((metric, index) => new { index, metric.RSquared })
                .OrderByDescending(_ => Math.Abs(_.RSquared.Mean))
                .Select(_ => _.index).ToList();
        }

        /// <summary>
        ///     FastForest: Выводит фичи в порядке уменьшения влияния на результат
        /// </summary>
        /// <param name="permutationCount"> Кол-во фич для расчета </param>
        public static List<int> FeaturePermutation(
            MLContext mlColntext,
            RegressionPredictionTransformer<FastForestRegressionModelParameters> model,
            IDataView preprocessedTrainData,
            int permutationCount)
        {
            var featuresPermutation = mlColntext.Regression.PermutationFeatureImportance(
                model,
                preprocessedTrainData,
                permutationCount: permutationCount);

            return featuresPermutation
                .Select((metric, index) => new { index, metric.RSquared })
                .OrderByDescending(_ => Math.Abs(_.RSquared.Mean))
                .Select(_ => _.index).ToList();
        }

        /// <summary>
        ///     LbfgsPoissonRegression: Выводит фичи в порядке уменьшения влияния на результат
        /// </summary>
        /// <param name="permutationCount"> Кол-во фич для расчета </param>
        public static List<int> FeaturePermutation(
            MLContext mlColntext,
            RegressionPredictionTransformer<PoissonRegressionModelParameters> model,
            IDataView preprocessedTrainData,
            int permutationCount)
        {
            var featuresPermutation = mlColntext.Regression.PermutationFeatureImportance(
                model,
                preprocessedTrainData,
                permutationCount: permutationCount);

            return featuresPermutation
                .Select((metric, index) => new { index, metric.RSquared })
                .OrderByDescending(_ => Math.Abs(_.RSquared.Mean))
                .Select(_ => _.index).ToList();
        }

        /// <summary>
        ///     FastTree: Выводит фичи в порядке уменьшения влияния на результат
        /// </summary>
        /// <param name="permutationCount"> Кол-во фич для расчета </param>
        public static List<int> FeaturePermutation(
            MLContext mlColntext,
            RegressionPredictionTransformer<FastTreeRegressionModelParameters> model,
            IDataView preprocessedTrainData,
            int permutationCount)
        {
            var featuresPermutation = mlColntext.Regression.PermutationFeatureImportance(
                model,
                preprocessedTrainData,
                permutationCount: permutationCount);

            return featuresPermutation
                .Select((metric, index) => new { index, metric.RSquared })
                .OrderByDescending(_ => Math.Abs(_.RSquared.Mean))
                .Select(_ => _.index).ToList();
        }

        /// <summary>
        ///     FastTreeTweedie: Выводит фичи в порядке уменьшения влияния на результат
        /// </summary>
        /// <param name="permutationCount"> Кол-во фич для расчета </param>
        public static List<int> FeaturePermutation(
            MLContext mlColntext,
            RegressionPredictionTransformer<FastTreeTweedieModelParameters> model,
            IDataView preprocessedTrainData,
            int permutationCount)
        {
            var featuresPermutation = mlColntext.Regression.PermutationFeatureImportance(
                model,
                preprocessedTrainData,
                permutationCount: permutationCount);

            return featuresPermutation
                .Select((metric, index) => new { index, metric.RSquared })
                .OrderByDescending(_ => Math.Abs(_.RSquared.Mean))
                .Select(_ => _.index).ToList();
        }

        /// <summary>
        ///     Gam: Выводит фичи в порядке уменьшения влияния на результат
        /// </summary>
        /// <param name="permutationCount"> Кол-во фич для расчета </param>
        public static List<int> FeaturePermutation(
            MLContext mlColntext,
            RegressionPredictionTransformer<GamRegressionModelParameters> model,
            IDataView preprocessedTrainData,
            int permutationCount)
        {
            var featuresPermutation = mlColntext.Regression.PermutationFeatureImportance(
                model,
                preprocessedTrainData,
                permutationCount: permutationCount);

            return featuresPermutation
                .Select((metric, index) => new { index, metric.RSquared })
                .OrderByDescending(_ => Math.Abs(_.RSquared.Mean))
                .Select(_ => _.index).ToList();
        }

        /// <summary>
        ///     Ols: Выводит фичи в порядке уменьшения влияния на результат
        /// </summary>
        /// <param name="permutationCount"> Кол-во фич для расчета </param>
        public static List<int> FeaturePermutation(
            MLContext mlColntext,
            RegressionPredictionTransformer<OlsModelParameters> model,
            IDataView preprocessedTrainData,
            int permutationCount)
        {
            var featuresPermutation = mlColntext.Regression.PermutationFeatureImportance(
                model,
                preprocessedTrainData,
                permutationCount: permutationCount);

            return featuresPermutation
                .Select((metric, index) => new { index, metric.RSquared })
                .OrderByDescending(_ => Math.Abs(_.RSquared.Mean))
                .Select(_ => _.index).ToList();
        }
    }
}
