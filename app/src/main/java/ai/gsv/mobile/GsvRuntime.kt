package ai.gsv.mobile

object GsvRuntime {
    val engine = TtsEngine(listOf(QnnHtpBackend(), CpuBackend()))
    val engineLock = Any()

    private var activityInstances = 0
    private var serviceInstances = 0

    @Volatile
    var apiEndpoint: String? = null
        internal set

    @Volatile
    var apiError: String? = null
        internal set

    fun retainActivity() = synchronized(engineLock) {
        activityInstances++
    }

    fun releaseActivity(finishing: Boolean) = synchronized(engineLock) {
        activityInstances = (activityInstances - 1).coerceAtLeast(0)
        if (finishing) closeEngineIfUnused()
    }

    fun retainService() = synchronized(engineLock) {
        serviceInstances++
    }

    fun releaseService() = synchronized(engineLock) {
        serviceInstances = (serviceInstances - 1).coerceAtLeast(0)
        closeEngineIfUnused()
    }

    private fun closeEngineIfUnused() {
        if (activityInstances == 0 && serviceInstances == 0) engine.close()
    }
}
