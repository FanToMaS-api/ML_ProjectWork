using System;
using Microsoft.ML;

namespace ML_ProjectWork.Helpers
{
    /// <summary>
    ///     Выводит предварительные расчеты на консоль
    /// </summary>
    internal static class DataPresentor
    {
        /// <summary>
        ///     Показывает какие данные парсит наша прога + приближенные предсказания
        /// </summary>
        public static void PeekDataViewInConsole(int countRowsToView, IDataView predictedDataView)
        {
            if (predictedDataView is null)
            {
                throw new ArgumentNullException("Fit model before using this method", nameof(predictedDataView));
            }

            var preViewTransformedData = predictedDataView.Preview(countRowsToView);
            foreach (var row in preViewTransformedData.RowView)
            {
                var columnCollection = row.Values;
                var lineToPrint = "Row--> ";
                foreach (var column in columnCollection)
                {
                    lineToPrint += $" | {column.Key}:{column.Value}";
                }

                Console.WriteLine(lineToPrint + "\n");
            }
        }
    }
}
