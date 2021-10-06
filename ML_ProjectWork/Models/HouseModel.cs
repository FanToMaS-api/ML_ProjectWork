using Microsoft.ML.Data;

namespace ML_ProjectWork
{
    /// <summary>
    ///     Модель дома
    /// </summary>
    internal class HouseModel
    {
        #region Properties

        /// <summary>
        ///     Id записи
        /// </summary>
        [LoadColumn(0)]
        public float Id { get; set; }

        /// <summary>
        ///     Дата создания записи
        /// </summary>
        [LoadColumn(1)]
        public string Date { get; set; }

        /// <summary>
        ///     Цена дома
        /// </summary>
        [LoadColumn(2)]
        public float Price { get; set; }

        /// <summary>
        ///     Кол-во спален
        /// </summary>
        [LoadColumn(3)]
        public float Bedrooms { get; set; }

        /// <summary>
        ///     Кол-во ванных комнат
        /// </summary>
        [LoadColumn(4)]
        public float Bathrooms { get; set; }

        /// <summary>
        ///     Размер жилой площади дома
        /// </summary>
        [LoadColumn(5)]
        public float LivingArea { get; set; }

        /// <summary>
        ///     Размер полной площади дома
        /// </summary>
        [LoadColumn(6)]
        public float Area { get; set; }

        /// <summary>
        ///     Кол-во этажей в доме
        /// </summary>
        [LoadColumn(7)]
        public float Floors { get; set; }

        /// <summary>
        ///     Есть ли выход к водоему
        /// </summary>
        [LoadColumn(8)]
        public float IsWaterFront { get; set; }

        /// <summary>
        ///     Оценка вида от 1 до 4
        /// </summary>
        [LoadColumn(9)]
        public float View { get; set; }

        #endregion
    }
}
