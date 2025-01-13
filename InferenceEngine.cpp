// ================================================================
// InferenceEngine.cpp
// ================================================================
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
    , mFormattedBuffer{}  // ここで明示的にコンストラクタ呼び出し
    , mPrevLen(0)
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

    // 1) Ensure mFormattedBuffer is sized to current n_ctx
    //  ここで mFormattedBuffer を llama_n_ctx(mCtx) 分だけ確保しておく
    const int nCtxTokens = llama_n_ctx(mCtx);
    if (mFormattedBuffer.size() < static_cast<size_t>(nCtxTokens)) {
        mFormattedBuffer.resize(nCtxTokens);
    }

    // Convert messages to llama format
    // メッセージをllama形式に変換
    std::vector<llama_chat_message> messagesForLlama = to_llama_messages(messages);

    // 2) Apply chat template
    //  チャットテンプレートを適用
    int newLen = llama_chat_apply_template(
        mModel,
        nullptr,
        messagesForLlama.data(),
        messagesForLlama.size(),
        /*force_system=*/true,
        mFormattedBuffer.data(),
        mFormattedBuffer.size()
        );

    if (newLen > static_cast<int>(mFormattedBuffer.size())) {
        //  万一 newLen が想定より大きければ再確保
        mFormattedBuffer.resize(newLen);
        newLen = llama_chat_apply_template(
            mModel,
            nullptr,
            messagesForLlama.data(),
            messagesForLlama.size(),
            /*force_system=*/true,
            mFormattedBuffer.data(),
            mFormattedBuffer.size()
            );
    }

    if (newLen < 0) {
        fprintf(stderr, "Failed to apply chat template.\n");
        return;
    }

    // 3) Build prompt string
    //  newLen, mPrevLen を使って substring
    std::string promptStd(
        mFormattedBuffer.begin() + mPrevLen,
        mFormattedBuffer.begin() + newLen
        );
    std::string response;

    // Tokenize the prompt
    // プロンプトをトークナイズ
    const int nPromptTokens = -llama_tokenize(
        mModel,
        promptStd.c_str(),
        promptStd.size(),
        nullptr,
        0,
        /*bos=*/true,
        /*newline=*/true
        );

    std::vector<llama_token> promptTokens(nPromptTokens);
    if (llama_tokenize(
            mModel,
            promptStd.c_str(),
            promptStd.size(),
            promptTokens.data(),
            promptTokens.size(),
            llama_get_kv_cache_used_cells(mCtx) == 0,
            /*newline=*/true) < 0)
    {
        emit generationError("failed to tokenize the prompt");
        return;
    }

    // Prepare a single batch for decoding
    llama_batch batch = llama_batch_get_one(promptTokens.data(), promptTokens.size());
    llama_token newTokenId;

    static constexpr int maxReplyTokens    {1024};
    static constexpr int extraCutoffTokens {32};
    int generatedTokenCount {0};

    // Decode until EOG
    while (true) {
        if (llama_decode(mCtx, batch)) {
            emit generationError("failed to decode");
            break;
        }

        // Sample next token
        newTokenId = llama_sampler_sample(mSampler, mCtx, /*last_n_tokens=*/-1);
        if (llama_token_is_eog(mModel, newTokenId)) {
            // End-of-generation
            break;
        }

        // Convert token -> piece
        char buf[256] = {};
        int n = llama_token_to_piece(mModel, newTokenId, buf, sizeof(buf), /*replaceNL=*/0, /*utf8=*/true);
        if (n < 0) {
            emit generationError("failed to convert token to piece");
            break;
        }

        std::string piece(buf, n);
        qDebug() << piece.c_str();

        response += piece;

        // Emit partial response
        emit partialResponseReady(QString::fromStdString(response));

        // Next batch
        batch = llama_batch_get_one(&newTokenId, 1);

        // Cut off if too long
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
        QCoreApplication::processEvents();
    }

    // Update mPrevLen for next usage
    mPrevLen = llama_chat_apply_template(
        mModel,
        nullptr,
        messagesForLlama.data(),
        messagesForLlama.size(),
        /*force_system=*/false,
        /*dst=*/nullptr,
        /*dst_size=*/0
        );
    if (mPrevLen < 0) {
        fprintf(stderr, "Failed to apply chat template.\n");
        mPrevLen = 0;  // fallback
    }

    // Emit final result
    emit generationFinished(QString::fromStdString(response));
}

/*
  remoteInitialized():
    - Returns the current state of mRemoteInitialized
*/
bool InferenceEngine::remoteInitialized() const
{
    return mRemoteInitialized;
}

/*
  setRemoteInitialized(newRemoteInitialized):
    - Updates mRemoteInitialized and emits remoteInitializedChanged if changed
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
    mSampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(mSampler, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(mSampler, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(mSampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    // Indicate successful init
    setRemoteInitialized(true);
    qDebug() << "Engine initialization complete.";
    qDebug() << "m_remoteInitialized =" << remoteInitialized();
}

/*
  reinitEngine():
    - Frees existing model/context/sampler
    - Resets remoteInitialized(false)
    - Calls do_engine_init() again
*/
void InferenceEngine::reinitEngine()
{
    qDebug() << "[reinitEngine] Re-initializing LLaMA engine...";

    // 1) Free existing sampler / ctx / model
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
    setRemoteInitialized(false);

    // 3) Rerun do_engine_init()
    do_engine_init();

    qDebug() << "[reinitEngine] Requested do_engine_init() again.";
}

// Default model path
const std::string InferenceEngine::mModelPath {
#ifdef LLAMA_MODEL_FILE
    LLAMA_MODEL_FILE
#else
#error "LLAMA_MODEL_FILE is not defined. Please define it via target_compile_definitions() in CMake."
#endif
};
