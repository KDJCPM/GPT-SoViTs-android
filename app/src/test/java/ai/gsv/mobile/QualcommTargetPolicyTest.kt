package ai.gsv.mobile

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test
import java.io.File

class QualcommTargetPolicyTest {
    @Test
    fun exactSm8750ArtifactIdentityPasses() {
        assertEquals(
            QualcommTargetSoc.SNAPDRAGON_8_ELITE,
            QualcommTargetPolicy.requireArtifactIdentity(model()),
        )
    }

    @Test
    fun everyProductTargetHasAnExactArtifactIdentity() {
        QualcommTargetSoc.entries.forEach { target ->
            val artifact = model().copy(
                targetSoc = target.id,
                targetAsic = target.asic,
                targetSocModel = target.socModel,
                htpArch = target.htpArch,
                supportedTargetSocs = setOf(target.id),
            )
            assertEquals(target, QualcommTargetPolicy.requireArtifactIdentity(artifact))
        }
    }

    @Test
    fun unknownTargetAndWrongHtpAreRejected() {
        assertThrows(IllegalArgumentException::class.java) {
            QualcommTargetPolicy.requireArtifactIdentity(model().copy(targetSoc = "snapdragon_8_gen_5"))
        }
        assertThrows(IllegalArgumentException::class.java) {
            QualcommTargetPolicy.requireArtifactIdentity(model().copy(htpArch = "V75"))
        }
    }

    @Test
    fun qairtAndRuntimeMustUseThePinnedSdkLine() {
        assertThrows(IllegalArgumentException::class.java) {
            QualcommTargetPolicy.requireArtifactIdentity(
                model().copy(qairtVersion = "2.42.0.251225"),
            )
        }
        assertThrows(IllegalArgumentException::class.java) {
            QualcommTargetPolicy.requireArtifactIdentity(
                model().copy(qnnRuntimeVersion = "2.42.0"),
            )
        }
    }

    @Test
    fun supportedTargetListMustBeExact() {
        assertThrows(IllegalArgumentException::class.java) {
            QualcommTargetPolicy.requireArtifactIdentity(model().copy(supportedTargetSocs = emptySet()))
        }
        assertThrows(IllegalArgumentException::class.java) {
            QualcommTargetPolicy.requireArtifactIdentity(
                model().copy(
                    supportedTargetSocs = setOf("snapdragon_8_elite", "snapdragon_8_gen_3"),
                ),
            )
        }
    }

    @Test
    fun platformMatchingAcceptsOnlyLeadingStableSocIdentifiers() {
        assertEquals(
            QualcommTargetSoc.SNAPDRAGON_8_ELITE,
            QualcommTargetSoc.fromPlatformStrings(listOf("SM8750-AB", "sun")),
        )
        assertEquals(
            QualcommTargetSoc.SNAPDRAGON_8_ELITE,
            QualcommTargetSoc.fromPlatformStrings(listOf("qcom", "SM8750")),
        )
        assertEquals(null, QualcommTargetSoc.fromPlatformStrings(listOf("sun", "fooSM8750")))
    }

    private fun model() = ModelPackage(
        root = File("."),
        name = "test",
        version = "v2ProPlus",
        sampleRate = 32_000,
        runtime = "qnn-htp",
        formatVersion = 1,
        executor = "qnn-htp",
        entrypoint = "synthesize_utf8_to_pcm16",
        deployable = true,
        targetSoc = "snapdragon_8_elite",
        targetSocFamily = QualcommTargetPolicy.FAMILY,
        targetAsic = "SM8750",
        targetSocModel = 69,
        htpArch = "V79",
        qairtVersion = "2.48.0.260626",
        qnnRuntimeVersion = QualcommTargetPolicy.RUNTIME_VERSION,
        backendArtifact = "runtime/qnn/executor.json",
        supportedTargetSocs = setOf("snapdragon_8_elite"),
    )
}
