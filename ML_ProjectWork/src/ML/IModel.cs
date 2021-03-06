using Microsoft.ML;
using ML_ProjectWork.ML.Enum;
using System.Collections.Generic;

namespace ML_ProjectWork.ML
{
    /// <summary>
    ///     Базовый интерфейс модели
    /// </summary>
    public interface IModel
    {
        #region Properties


        /// <summary>
        ///     Абсолютная ошибки модели
        /// </summary>
        public double MeanAbsoluteError { get; }

        /// <summary>
        ///     Коэффициент детерминации (показывает насколько близки наши данные к прямой)
        /// </summary>
        public double RSquared { get; }

        /// <summary>
        ///     Корень из среднеквадратичной ошибки модели
        /// </summary>
        public double RootMeanSquaredError { get; }

        /// <summary>
        ///     Список имен фичей
        /// </summary>
        public List<string> Features { get; }

        /// <summary>
        ///     Обученная модель
        /// </summary>
        public ITransformer Model { get; }

        /// <summary>
        ///     MlContext
        /// </summary>
        public MLContext MlContext { get; }

        /// <summary>
        ///     Тип тренера
        /// </summary>
        public TrainerModel Trainer { get; }

        /// <summary>
        ///     Исходные данные с предсказаниями
        /// </summary>
        public IDataView PredictedData { get; }

        /// <summary>
        ///     Данные для тренировки
        /// </summary>
        public IDataView TrainData { get; }

        /// <summary>
        ///     Данные для тестирования
        /// </summary>
        public IDataView TestData { get; }

        /// <summary>
        ///     Определяет, загружать ли сохраненную модель
        /// </summary>
        public bool IsLoadSavedModel { get; }

        #endregion

        #region Public methods

        /// <summary>
        ///     Проводит обучение модели
        /// </summary>
        public void Fit();

        /// <summary>
        ///     Устанавливает тренера в соответствии с выбранным Trainer
        /// </summary>
        public IEstimator<ITransformer> SetTrainer();

        /// <summary>
        ///     Сохраняет модель
        /// </summary>
        public void Save();

        #endregion
    }
}
