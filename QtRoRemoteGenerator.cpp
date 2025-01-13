#include "QtRoRemoteGenerator.h"

/*
  QtRORemoteGenerator constructor:
    - Connects engine signals to the corresponding signals/slots in this class
    - Allows the remote interface to observe engine state via inherited properties
  QtRORemoteGeneratorのコンストラクタ:
    - エンジンのシグナルをこのクラスのシグナル/スロットに接続
    - 継承したプロパティを介して、リモート側がエンジンの状態を把握できるようにする
*/
QtRORemoteGenerator::QtRORemoteGenerator(QObject *parent)
    : LlamaResponseGeneratorSimpleSource{parent}
{
    // When InferenceEngine reinitialized -> reinitialized signal here
    // InferenceEngineが再初期化されたら -> このクラスのreinitializedシグナルをemit
    connect(&mInferenceEngine, &InferenceEngine::reinitialized,
            this, &QtRORemoteGenerator::reinitialized);

    // Partial/final response
    // 部分/最終レスポンスを受け取り、このクラスのシグナルに渡す
    connect(&mInferenceEngine, &InferenceEngine::partialResponseReady,
            this, &QtRORemoteGenerator::partialResponseReady);
    connect(&mInferenceEngine, &InferenceEngine::generationFinished,
            this, &QtRORemoteGenerator::generationFinished);

    // Error reporting
    // エラー報告を受け取り、このクラスのシグナルに渡す
    connect(&mInferenceEngine, &InferenceEngine::generationError,
            this, &QtRORemoteGenerator::generationError);

    // Remote initialization state
    // リモート初期化状態が変化したら、setRemoteInitializedを呼び出し
    connect(&mInferenceEngine, &InferenceEngine::remoteInitializedChanged,
            this, &QtRORemoteGenerator::setRemoteInitialized);
}

/*
  generate(messages):
    - Delegates text generation to the internal InferenceEngine
  generate(messages):
    - 内部のInferenceEngineにテキスト生成を委譲
*/
void QtRORemoteGenerator::generate(const QList<LlamaChatMessage> &messages)
{
    mInferenceEngine.generate(messages);
}

/*
  reinitEngine():
    - Re-initializes the internal InferenceEngine
  reinitEngine():
    - 内部のInferenceEngineを再初期化
*/
void QtRORemoteGenerator::reinitEngine()
{
    mInferenceEngine.reinitEngine();
}
