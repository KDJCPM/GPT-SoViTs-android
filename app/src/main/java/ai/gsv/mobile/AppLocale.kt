package ai.gsv.mobile

import android.content.Context
import android.content.res.Configuration
import java.util.Locale

enum class AppLanguage(val tag: String) {
    CHINESE("zh"),
    ENGLISH("en"),
}

object AppLocale {
    private const val PREFERENCES = "settings"
    private const val LANGUAGE = "app_language"

    fun selected(context: Context): AppLanguage {
        val stored = context.getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)
            .getString(LANGUAGE, null)
        return AppLanguage.entries.firstOrNull { it.tag == stored } ?: systemDefault(context)
    }

    fun wrap(context: Context): Context {
        val locale = Locale.forLanguageTag(selected(context).tag)
        val configuration = Configuration(context.resources.configuration)
        configuration.setLocale(locale)
        configuration.setLayoutDirection(locale)
        return context.createConfigurationContext(configuration)
    }

    fun set(context: Context, language: AppLanguage) {
        context.getSharedPreferences(PREFERENCES, Context.MODE_PRIVATE)
            .edit()
            .putString(LANGUAGE, language.tag)
            .apply()
    }

    private fun systemDefault(context: Context): AppLanguage {
        val locales = context.resources.configuration.locales
        val primary = if (!locales.isEmpty) locales[0] else Locale.getDefault()
        return if (primary.language.equals("zh", ignoreCase = true)) {
            AppLanguage.CHINESE
        } else {
            AppLanguage.ENGLISH
        }
    }
}
