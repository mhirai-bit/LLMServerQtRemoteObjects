#include "InferenceEngine.h"
#include <QDebug>
#include <QThreadPool>

/*
  Constructor:
    - Optionally receives a llama_model and llama_context
    - Spawns a thread to handle do_engine_init() asynchronously
  コンストラクタ:
    - llama_model, llama_contextを任意で受け取れる
    - do_engine_init()を非同期実行するスレッドを開始
*/
InferenceEngine::InferenceEngine(QObject *parent,
                                 llama_model *model,
                                 llama_context *ctx)
    : QObject(parent)
    , mModel(model)
    , mCtx(ctx)
{
    // Start initialization in background
    // バックグラウンドで初期化開始
    QThreadPool::globalInstance()->start([this]() {
        do_engine_init();
    });
}

/*
  Destructor:
    - If mSampler is allocated, free it
  デストラクタ:
    - mSamplerが割り当てられていれば解放する
*/
InferenceEngine::~InferenceEngine()
{
    if (mSampler) {
        llama_sampler_free(mSampler);
        mSampler = nullptr;
    }
}

/*
  generate(messages):
    - Tokenizes and decodes user messages to produce text
    - Emits partial/final responses during generation
  generate(messages):
    - ユーザーメッセージをトークナイズ/デコードしてテキスト生成
    - 推論過程で部分/最終レスポンスをemitする
*/
void InferenceEngine::generate(const QList<LlamaChatMessage>& messages)
{
    qDebug() << "Generating response...";

    static std::vector<char> formatted(llama_n_ctx(mCtx));
    static int prevLen {0};

    // Convert messages to llama format
    // メッセージをllama形式に変換
    std::vector<llama_chat_message> messagesForLlama = to_llama_messages(messages);

    // Apply chat template
    // チャットテンプレート適用
    int newLen = llama_chat_apply_template(
        mModel,
        nullptr,
        messagesForLlama.data(),
        messagesForLlama.size(),
        true,
        formatted.data(),
        formatted.size()
        );
    if (newLen > static_cast<int>(formatted.size())) {
        formatted.resize(newLen);
        newLen = llama_chat_apply_template(
            mModel,
            nullptr,
            messagesForLlama.data(),
            messagesForLlama.size(),
            true,
            formatted.data(),
            formatted.size()
            );
    }
    if (newLen < 0) {
        fprintf(stderr, "Failed to apply chat template.\n");
        return;
    }

    std::string promptStd(formatted.begin() + prevLen, formatted.begin() + newLen);
    std::string response;

    // Tokenize the prompt
    // プロンプトをトークナイズ
    const int nPromptTokens = -llama_tokenize(
        mModel,
        promptStd.c_str(),
        promptStd.size(),
        nullptr,
        0,
        true,
        true
        );

    std::vector<llama_token> promptTokens(nPromptTokens);
    if (llama_tokenize(
            mModel,
            promptStd.c_str(),
            promptStd.size(),
            promptTokens.data(),
            promptTokens.size(),
            llama_get_kv_cache_used_cells(mCtx) == 0,
            true) < 0)
    {
        emit generationError("failed to tokenize the prompt");
        return;
    }

    // Prepare a single batch for decoding
    // 1バッチでデコードを行う準備
    llama_batch batch = llama_batch_get_one(promptTokens.data(), promptTokens.size());
    llama_token newTokenId;

    static constexpr int maxReplyTokens    {1024};
    static constexpr int extraCutoffTokens {32};
    int generatedTokenCount {0};

    // Decode until EOG
    // 終了トークンまでデコードする
    while (true) {
        const int nCtxUsed = llama_get_kv_cache_used_cells(mCtx);
        if (llama_decode(mCtx, batch)) {
            emit generationError("failed to decode");
            break;
        }

        // Sample next token
        // 次トークンをサンプリング
        newTokenId = llama_sampler_sample(mSampler, mCtx, -1);

        if (llama_token_is_eog(mModel, newTokenId)) {
            // End-of-generation
            // 生成終了
            break;
        }

        char buf[256] = {};
        int n = llama_token_to_piece(mModel, newTokenId, buf, sizeof(buf), 0, true);
        if (n < 0) {
            emit generationError("failed to convert token to piece");
            break;
        }

        std::string piece(buf, n);
        qDebug() << piece.c_str(); // or just piece if capturing

        response += piece;

        // Emit partial response
        // 部分的なレスポンスをemit
        emit partialResponseReady(QString::fromStdString(response));

        // Next batch
        // 次バッチを作成
        batch = llama_batch_get_one(&newTokenId, 1);

        // Cut off if too long
        // レスポンスが長すぎる場合は打ち切り
        ++generatedTokenCount;
        if (generatedTokenCount > maxReplyTokens) {
            if (piece.find('\n') != std::string::npos) {
                qDebug() << "Cutting off at newline.";
                break;
            } else if (generatedTokenCount > maxReplyTokens + extraCutoffTokens) {
                qDebug() << "Cutting off after extra tokens.";
                break;
            }
        }

        // Process events for immediate partial updates
        // 部分更新を即時反映するためにイベント処理
        QCoreApplication::processEvents();
    }

    // Update prevLen for next usage
    // 次回呼び出しに備えてprevLenを更新
    prevLen = llama_chat_apply_template(
        mModel,
        nullptr,
        messagesForLlama.data(),
        messagesForLlama.size(),
        false,
        nullptr,
        0
        );
    if (prevLen < 0) {
        fprintf(stderr, "Failed to apply chat template.\n");
    }

    // Emit final result
    // 最終結果を通知
    emit generationFinished(QString::fromStdString(response));
}

/*
  remoteInitialized():
    - Returns the current state of mRemoteInitialized
  remoteInitialized():
    - mRemoteInitializedの現在の状態を返す
*/
bool InferenceEngine::remoteInitialized() const
{
    return mRemoteInitialized;
}

/*
  setRemoteInitialized(newRemoteInitialized):
    - Updates mRemoteInitialized and emits remoteInitializedChanged if changed
  setRemoteInitialized(newRemoteInitialized):
    - mRemoteInitializedを更新し、変化があればremoteInitializedChangedをemit
*/
void InferenceEngine::setRemoteInitialized(bool newRemoteInitialized)
{
    if (mRemoteInitialized == newRemoteInitialized)
        return;
    mRemoteInitialized = newRemoteInitialized;
    emit remoteInitializedChanged(mRemoteInitialized);
}

/*
  to_llama_messages(userMessages):
    - Converts QList<LlamaChatMessage> to std::vector<llama_chat_message>
  to_llama_messages(userMessages):
    - QList<LlamaChatMessage>をstd::vector<llama_chat_message>に変換
*/
std::vector<llama_chat_message>
InferenceEngine::to_llama_messages(const QList<LlamaChatMessage> &userMessages)
{
    std::vector<llama_chat_message> llamaMessages;
    llamaMessages.reserve(userMessages.size());

    for (const auto &um : userMessages) {
        llama_chat_message lm;
        lm.role    = strdup(um.role().toUtf8().constData());
        lm.content = strdup(um.content().toUtf8().constData());
        llamaMessages.push_back(lm);
    }
    return llamaMessages;
}

/*
  do_engine_init():
    - Loads the model and context in a background thread
    - Initializes the sampler
    - Sets remoteInitialized(true) on success
  do_engine_init():
    - バックグラウンドでモデルとコンテキストをロード
    - サンプラーを初期化
    - 成功時にremoteInitialized(true)をセット
*/
void InferenceEngine::do_engine_init()
{
    ggml_backend_load_all();

    mModelParams = llama_model_default_params();
    mModelParams.n_gpu_layers = mNGl;

    mModel = llama_load_model_from_file(mModelPath.c_str(), mModelParams);
    if (!mModel) {
        fprintf(stderr, "Error: unable to load model.\n");
        return;
    }

    mCtxParams = llama_context_default_params();
    mCtxParams.n_ctx   = mNCtx;
    mCtxParams.n_batch = mNCtx;

    mCtx = llama_new_context_with_model(mModel, mCtxParams);
    if (!mCtx) {
        fprintf(stderr, "Error: failed to create llama_context.\n");
        return;
    }

    // Initialize sampler
    // サンプラーの初期化
    mSampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(mSampler, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(mSampler, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(mSampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    // Indicate successful init
    // 初期化成功を通知
    setRemoteInitialized(true);
    qDebug() << "Engine initialization complete.";
    qDebug() << "m_remoteInitialized =" << remoteInitialized();
}

/*
  reinitEngine():
    - Frees existing model/context/sampler
    - Resets remoteInitialized(false)
    - Calls do_engine_init() again
  reinitEngine():
    - 既存のモデル/コンテキスト/サンプラーを解放
    - remoteInitialized(false)にリセット
    - do_engine_init()を再度呼び出し
*/
void InferenceEngine::reinitEngine()
{
    qDebug() << "[reinitEngine] Re-initializing LLaMA engine...";

    // 1) Free existing sampler / ctx / model
    // 1) 既存のsampler / ctx / modelを解放
    if (mSampler) {
        llama_sampler_free(mSampler);
        mSampler = nullptr;
    }

    if (mCtx) {
        llama_free(mCtx);
        mCtx = nullptr;
    }

    if (mModel) {
        llama_free_model(mModel);
        mModel = nullptr;
    }

    // 2) Reset remoteInitialized to false
    // 2) remoteInitialized を false にする
    setRemoteInitialized(false);

    // 3) Rerun do_engine_init()
    // 3) do_engine_init()を再実行
    do_engine_init();

    qDebug() << "[reinitEngine] Requested do_engine_init() again.";
}

/*
  mModelPath:
    - Defined via target_compile_definitions in CMake
    - Provide a default if not specified
  mModelPath:
    - CMakeのtarget_compile_definitionsで定義
    - 指定がない場合はエラー
*/
const std::string InferenceEngine::mModelPath {
#ifdef LLAMA_MODEL_FILE
    LLAMA_MODEL_FILE
#else
#error "LLAMA_MODEL_FILE is not defined. Please define it via target_compile_definitions() in CMake."
#endif
};
