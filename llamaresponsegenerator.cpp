#include "llamaresponsegenerator.h"
#include <QDebug>
#include <QThreadPool>

// Constructor: store model/context & start do_engine_init in background
// コンストラクタ: model/contextを保存し、バックグラウンドでdo_engine_initを開始
LlamaResponseGenerator::LlamaResponseGenerator(QObject *parent,
                                               llama_model *model,
                                               llama_context *ctx)
    : LlamaResponseGeneratorSimpleSource(parent)
    , mModel(model)
    , mCtx(ctx)
{
    QThreadPool::globalInstance()->start([this]() {
        do_engine_init();
    });
}

// Destructor: release sampler if it exists
// デストラクタ: samplerがあれば解放
LlamaResponseGenerator::~LlamaResponseGenerator()
{
    if (mSampler) {
        llama_sampler_free(mSampler);
        mSampler = nullptr;
    }
}

// generate(...): tokenize, decode, emit partial/final
// generate(...): トークナイズ、デコード、途中／最終をemit
void LlamaResponseGenerator::generate(const QList<LlamaChatMessage>& messages)
{
    static std::vector<char> formatted(llama_n_ctx(mCtx));
    static int prevLen {0};

    // Convert to llama messages
    // llama用のメッセージ形式に変換
    std::vector<llama_chat_message> messagesForLlama = to_llama_messages(messages);

    // Apply chat template
    // チャットテンプレートを適用
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

    // Tokenize prompt
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

    // Prepare single batch for decoding
    // デコード用のバッチを準備
    llama_batch batch = llama_batch_get_one(promptTokens.data(), promptTokens.size());
    llama_token newTokenId;

    static constexpr int maxReplyTokens    {1024};
    static constexpr int extraCutoffTokens {32};
    int generatedTokenCount {0};

    // Decode until EOG
    // 終了トークンに達するまでデコード
    while (true) {
        const int nCtxUsed = llama_get_kv_cache_used_cells(mCtx);
        if (llama_decode(mCtx, batch)) {
            emit generationError("failed to decode");
            break;
        }

        // Sample next token
        // 次のトークンをサンプリング
        newTokenId = llama_sampler_sample(mSampler, mCtx, -1);

        if (llama_token_is_eog(mModel, newTokenId)) {
            // End-of-generation
            // 終了トークン
            break;
        }

        char buf[256] = {};
        int n = llama_token_to_piece(mModel, newTokenId, buf, sizeof(buf), 0, true);
        if (n < 0) {
            emit generationError("failed to convert token to piece");
            break;
        }

        std::string piece(buf, n);
        response += piece;

        // Emit partial progress
        // 部分的なレスポンスを通知
        emit partialResponseReady(QString::fromStdString(response));

        // Next batch
        // 次のバッチを用意
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

        // For immediate partial updates, we can process events
        QCoreApplication::processEvents();
    }

    // Update prevLen
    // 次の呼び出しに備えて prevLen を更新
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
    // 最終的な結果を通知
    emit generationFinished(QString::fromStdString(response));
}

// to_llama_messages(...): convert from QList to std::vector
// to_llama_messages(...): QList -> std::vector 変換
std::vector<llama_chat_message>
LlamaResponseGenerator::to_llama_messages(const QList<LlamaChatMessage> &userMessages)
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

// do_engine_init(): loads model/context in background
// do_engine_init(): バックグラウンドでモデル/コンテキストをロード
void LlamaResponseGenerator::do_engine_init()
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

    // Notify remote side that init succeeded
    // リモート側に初期化成功を通知
    setRemoteInitialized(true);
    qDebug() << "Engine initialization complete.";
    qDebug() << "m_remoteInitialized =" << remoteInitialized();
}

void LlamaResponseGenerator::reinitEngine()
{
    qDebug() << "[reinitEngine] Re-initializing LLaMA engine...";

    // 1) 既存の sampler / ctx / model を解放
    //---------------------------------------------------
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

    // 2) remoteInitialized フラグをいったん下げる
    //    （再度成功するまでfalseにしておき、クライアント側にもリセットが伝わる）
    //---------------------------------------------------
    setRemoteInitialized(false);

    // 3) do_engine_init()を再度走らせる
    //---------------------------------------------------
    do_engine_init();

    qDebug() << "[reinitEngine] Requested do_engine_init() again.";
}

// Default model path (CMake)
const std::string LlamaResponseGenerator::mModelPath {
#ifdef LLAMA_MODEL_FILE
    LLAMA_MODEL_FILE
#else
#error "LLAMA_MODEL_FILE is not defined. Please define it via target_compile_definitions() in CMake."
#endif
};
