namespace ML_ProjectWork.ML
{
    /// <summary>
    ///     Определяет модели тренеров
    /// </summary>
    public enum TrainerModel
    {
        /// <summary>
        ///     LbfgsPoissonRegression
        /// </summary>
        LbfgsPoissonRegression,

        /// <summary>
        ///     FastForest
        /// </summary>
        FastForest,

        /// <summary>
        ///     FastTree
        /// </summary>
        FastTree,

        /// <summary>
        ///     FastTreeTweedie
        /// </summary>
        FastTreeTweedie,

        /// <summary>
        ///     Gam
        /// </summary>
        Gam,

        /// <summary>
        ///     LightGbm
        /// </summary>
        LightGbm,

        /// <summary>
        ///     Ols
        /// </summary>
        Ols,

        /// <summary>
        ///     OnlineGradientDescent
        /// </summary>
        OnlineGradientDescent,

        /// <summary>
        ///     Sdca
        /// </summary>
        Sdca,

        /// <summary>
        ///     Без тренера (для графиков)
        /// </summary>
        None
    }
}
