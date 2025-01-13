#include "QtRoRemoteGenerator.h"

QtRORemoteGenerator::QtRORemoteGenerator(QObject *parent)
    : LlamaResponseGeneratorSimpleSource{parent}
{
    connect(&mInferenceEngine, &InferenceEngine::reinitialized,
            this, &QtRORemoteGenerator::reinitialized);
    connect(&mInferenceEngine, &InferenceEngine::partialResponseReady,
            this, &QtRORemoteGenerator::partialResponseReady);
    connect(&mInferenceEngine, &InferenceEngine::generationFinished,
            this, &QtRORemoteGenerator::generationFinished);
    connect(&mInferenceEngine, &InferenceEngine::generationError,
            this, &QtRORemoteGenerator::generationError);
    connect(&mInferenceEngine, &InferenceEngine::remoteInitializedChanged,
            this, &QtRORemoteGenerator::setRemoteInitialized);
}

void QtRORemoteGenerator::generate(const QList<LlamaChatMessage> &messages)
{
    mInferenceEngine.generate(messages);
}

void QtRORemoteGenerator::reinitEngine()
{
    mInferenceEngine.reinitEngine();
}
