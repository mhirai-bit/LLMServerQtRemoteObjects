#ifndef INFERENCEENGINE_H
#define INFERENCEENGINE_H

#include "rep_LlamaResponseGenerator_source.h"  // Short definitions from .rep file / .repファイルからの定義
#include "llama.h"
#include <QObject>
#include <QString>

class InferenceEngine : public QObject
{
    Q_OBJECT
    Q_PROPERTY(bool remoteInitialized READ remoteInitialized WRITE setRemoteInitialized NOTIFY remoteInitializedChanged FINAL)
public:
    explicit InferenceEngine(QObject *parent = nullptr,
                             llama_model *model = nullptr,
                             llama_context *ctx = nullptr);
    ~InferenceEngine() override;

    void generate(const QList<LlamaChatMessage>& messages);
    void reinitEngine();

    bool remoteInitialized() const;
    void setRemoteInitialized(bool newRemoteInitialized);

signals:
    void reinitialized();
    void partialResponseReady(const QString &response);
    void generationFinished(const QString &response);
    void generationError(const QString &error);

    void remoteInitializedChanged(bool newRemoteInitialized);

private:
    // Internal parameters
    // 内部パラメータ
    static constexpr int mNGl  {99};
    static constexpr int mNCtx {2048};

    // Default model path (via CMake)
    // デフォルトのモデルパス (CMakeで定義)
    static const std::string mModelPath;

    // Holds llama params/context/model/sampler
    // llama 用パラメータ／コンテキスト／モデル／サンプラーを保持
    llama_model_params mModelParams;
    llama_sampler*     mSampler    {nullptr};
    llama_model*       mModel      {nullptr};
    llama_context_params mCtxParams;
    llama_context*       mCtx      {nullptr};

    bool mRemoteInitialized {false};

    // Runs heavy init in separate thread
    // 別スレッドでの重い初期化処理
    void do_engine_init();

    // Convert QList<LlamaChatMessage> -> std::vector<llama_chat_message>
    // QList<LlamaChatMessage>からstd::vector<llama_chat_message>に変換
    std::vector<llama_chat_message> to_llama_messages(const QList<LlamaChatMessage> &userMessages);
};

#endif // INFERENCEENGINE_H
