using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Trainers.LightGbm;

namespace ML_ProjectWork
{
    /// <summary>
    ///     Вычисляет значимость фич
    /// </summary>
    internal static class FeatureHelper
    {
        /// <summary>
        ///     Выводит фичи в порядке уменьшения влияния на результат
        /// </summary>
        public static List<int> FeaturePermutation(
            MLContext mlColntext,
            RegressionPredictionTransformer<LinearModelParameters> model,
            IDataView preprocessedTrainData,
            string[] featureColumnNames)
        {
            var featuresPermutation = mlColntext.Regression.PermutationFeatureImportance(
                model,
                preprocessedTrainData,
                permutationCount: 5);

            return featuresPermutation
                .Select((metric, index) => new { index, metric.RSquared })
                .OrderByDescending(_ => Math.Abs(_.RSquared.Mean))
                .Select(_ => _.index).ToList();
        }

        /// <summary>
        ///     Выводит фичи в порядке уменьшения влияния на результат
        /// </summary>
        public static List<int> FeaturePermutation(
            MLContext mlColntext, 
            RegressionPredictionTransformer<LinearRegressionModelParameters> model,
            IDataView preprocessedTrainData, 
            string[] featureColumnNames)
        {
            var featuresPermutation = mlColntext.Regression.PermutationFeatureImportance(
                model,
                preprocessedTrainData,
                permutationCount: 5);

            return featuresPermutation
                .Select((metric, index) => new { index, metric.RSquared })
                .OrderByDescending(_ => Math.Abs(_.RSquared.Mean))
                .Select(_ => _.index).ToList();
        }

        public static List<int> FeaturePermutation(
            MLContext mlColntext, 
            RegressionPredictionTransformer<LightGbmRegressionModelParameters> model,
            IDataView preprocessedTrainData, 
            string[] featureColumnNames)
        {
            var featuresPermutation = mlColntext.Regression.PermutationFeatureImportance(
                model,
                preprocessedTrainData,
                permutationCount: 5);

            return featuresPermutation
                .Select((metric, index) => new { index, metric.RSquared })
                .OrderByDescending(_ => Math.Abs(_.RSquared.Mean))
                .Select(_ => _.index).ToList();
        }
    }
}
