package ai.gsv.mobile

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.IBinder
import android.os.ResultReceiver
import fi.iki.elonen.NanoHTTPD
import java.io.File

class LocalOpenAiService : Service() {
    private var server: LocalOpenAiServer? = null

    override fun onCreate() {
        super.onCreate()
        GsvRuntime.retainService()
    }

    override fun attachBaseContext(newBase: Context) {
        super.attachBaseContext(AppLocale.wrap(newBase))
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            stopSelf()
            return START_NOT_STICKY
        }
        val port = intent?.getIntExtra(EXTRA_PORT, -1) ?: -1
        val receiver = intent?.resultReceiver()
        if (port !in 1024..65535) {
            receiver?.send(RESULT_ERROR, android.os.Bundle().apply {
                putString(EXTRA_ERROR, getString(R.string.api_port_invalid))
            })
            stopSelf()
            return START_NOT_STICKY
        }
        createNotificationChannel()
        startForeground(NOTIFICATION_ID, notification(port))
        runCatching {
            server?.stop()
            LocalOpenAiServer(
                port,
                File(cacheDir, "openai-api"),
                { GsvRuntime.engine.isLoaded },
                { GsvRuntime.engine.backendName },
                { GsvRuntime.engine.referenceExactPcm16kSamples },
                { request, output ->
                    synchronized(GsvRuntime.engineLock) {
                        GsvRuntime.engine.synthesize(request, output)
                    }
                },
                requestLock = GsvRuntime.engineLock,
            ).also { it.start(NanoHTTPD.SOCKET_READ_TIMEOUT, false) }
        }.onSuccess { running ->
            server = running
            GsvRuntime.apiEndpoint = running.endpoint
            GsvRuntime.apiError = null
            receiver?.send(RESULT_STARTED, android.os.Bundle().apply {
                putString(EXTRA_ENDPOINT, running.endpoint)
            })
        }.onFailure { error ->
            GsvRuntime.apiEndpoint = null
            GsvRuntime.apiError = error.message ?: error::class.java.simpleName
            receiver?.send(RESULT_ERROR, android.os.Bundle().apply {
                putString(EXTRA_ERROR, GsvRuntime.apiError)
            })
            stopSelf()
        }
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        server?.stop()
        server = null
        GsvRuntime.apiEndpoint = null
        stopForeground(STOP_FOREGROUND_REMOVE)
        GsvRuntime.releaseService()
        super.onDestroy()
    }

    private fun createNotificationChannel() {
        val manager = getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(
            NotificationChannel(
                CHANNEL_ID,
                getString(R.string.api_notification_channel),
                NotificationManager.IMPORTANCE_LOW,
            )
        )
    }

    private fun notification(port: Int): Notification {
        val launch = PendingIntent.getActivity(
            this,
            0,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
        )
        return Notification.Builder(this, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.stat_sys_upload_done)
            .setContentTitle(getString(R.string.api_notification_title))
            .setContentText(getString(R.string.api_notification_text, port))
            .setContentIntent(launch)
            .setOngoing(true)
            .build()
    }

    @Suppress("DEPRECATION")
    private fun Intent.resultReceiver(): ResultReceiver? =
        if (android.os.Build.VERSION.SDK_INT >= 33) {
            getParcelableExtra(EXTRA_RECEIVER, ResultReceiver::class.java)
        } else {
            getParcelableExtra(EXTRA_RECEIVER)
        }

    companion object {
        const val ACTION_START = "ai.gsv.mobile.action.START_OPENAI_API"
        const val ACTION_STOP = "ai.gsv.mobile.action.STOP_OPENAI_API"
        const val EXTRA_PORT = "port"
        const val EXTRA_RECEIVER = "receiver"
        const val EXTRA_ENDPOINT = "endpoint"
        const val EXTRA_ERROR = "error"
        const val RESULT_STARTED = 1
        const val RESULT_ERROR = 2
        private const val CHANNEL_ID = "openai-local-api"
        private const val NOTIFICATION_ID = 9880

        fun startIntent(context: Context, port: Int, receiver: ResultReceiver): Intent =
            Intent(context, LocalOpenAiService::class.java)
                .setAction(ACTION_START)
                .putExtra(EXTRA_PORT, port)
                .putExtra(EXTRA_RECEIVER, receiver)

        fun stopIntent(context: Context): Intent =
            Intent(context, LocalOpenAiService::class.java).setAction(ACTION_STOP)
    }
}
