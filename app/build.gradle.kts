plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

val acceptanceAbi = providers.gradleProperty("acceptanceAbi").orNull

android {
    namespace = "ai.gsv.mobile"
    compileSdk = 34

    defaultConfig {
        applicationId = "ai.gsv.mobile"
        minSdk = 26
        targetSdk = 34
        versionCode = 5
        versionName = "0.1.4"
        if (acceptanceAbi != null) {
            ndk { abiFilters += acceptanceAbi }
        }
    }

    buildFeatures { compose = true; buildConfig = true }
    packaging {
        jniLibs { useLegacyPackaging = true }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.activity:activity-compose:1.9.3")
    implementation(platform("androidx.compose:compose-bom:2024.09.03"))
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material:material-icons-core")
    implementation("androidx.compose.material:material-icons-extended")
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-tooling-preview")
    debugImplementation("androidx.compose.ui:ui-tooling")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.7")
    implementation("org.nanohttpd:nanohttpd:2.3.1")
    implementation("org.pytorch:pytorch_android:2.1.0")
    // QNN is the first-class ONNX Runtime build for the NPU path.  It brings the
    // QNN execution provider; qnn-runtime is pulled transitively by this artifact.
    implementation("com.microsoft.onnxruntime:onnxruntime-android-qnn:1.28.0")
}
