#include "llamaresponsegenerator.h"
#include <QDebug>
#include <QtConcurrent>

// Constructor: store model/context & start do_engine_init in background
// コンストラクタ: model/contextを保存し、バックグラウンドでdo_engine_initを開始
LlamaResponseGenerator::LlamaResponseGenerator(QObject *parent,
                                               llama_model *model,
                                               llama_context *ctx)
    : LlamaResponseGeneratorSimpleSource(parent)
    , m_model(model)
    , m_ctx(ctx)
{
    QtConcurrent::run(&LlamaResponseGenerator::do_engine_init, this);
}

// Destructor: release sampler if it exists
// デストラクタ: samplerがあれば解放
LlamaResponseGenerator::~LlamaResponseGenerator()
{
    if (m_sampler) {
        llama_sampler_free(m_sampler);
        m_sampler = nullptr;
    }
}

// generate(...): tokenize, decode, emit partial/final
// generate(...): トークナイズ、デコード、途中／最終をemit
void LlamaResponseGenerator::generate(const QList<LlamaChatMessage>& messages)
{
    static std::vector<char> formatted(llama_n_ctx(m_ctx));
    static int prev_len {0};

    // Convert to llama messages
    // llama用のメッセージ形式に変換
    std::vector<llama_chat_message> messages_for_llama = to_llama_messages(messages);

    // Apply chat template
    // チャットテンプレートを適用
    int new_len = llama_chat_apply_template(
        m_model,
        nullptr,
        messages_for_llama.data(),
        messages_for_llama.size(),
        true,
        formatted.data(),
        formatted.size()
        );
    if (new_len > static_cast<int>(formatted.size())) {
        formatted.resize(new_len);
        new_len = llama_chat_apply_template(
            m_model,
            nullptr,
            messages_for_llama.data(),
            messages_for_llama.size(),
            true,
            formatted.data(),
            formatted.size()
            );
    }
    if (new_len < 0) {
        fprintf(stderr, "Failed to apply chat template.\n");
        return;
    }

    std::string prompt_std(formatted.begin() + prev_len, formatted.begin() + new_len);
    std::string response;

    // Tokenize prompt
    // プロンプトをトークナイズ
    const int n_prompt_tokens = -llama_tokenize(
        m_model,
        prompt_std.c_str(),
        prompt_std.size(),
        nullptr,
        0,
        true,
        true
        );

    std::vector<llama_token> prompt_tokens(n_prompt_tokens);
    if (llama_tokenize(
            m_model,
            prompt_std.c_str(),
            prompt_std.size(),
            prompt_tokens.data(),
            prompt_tokens.size(),
            llama_get_kv_cache_used_cells(m_ctx) == 0,
            true) < 0)
    {
        emit generationError("failed to tokenize the prompt");
    }

    // Prepare single batch for decoding
    // デコード用のバッチを準備
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    llama_token new_token_id;

    static constexpr int max_reply_tokens    {1024};
    static constexpr int extra_cutoff_tokens {32};
    int generated_token_count {0};

    // Decode until EOG
    // 終了トークンに達するまでデコード
    while (true) {
        const int n_ctx_used = llama_get_kv_cache_used_cells(m_ctx);
        if (llama_decode(m_ctx, batch)) {
            emit generationError("failed to decode");
            break;
        }

        // Sample next token
        // 次のトークンをサンプリング
        new_token_id = llama_sampler_sample(m_sampler, m_ctx, -1);

        if (llama_token_is_eog(m_model, new_token_id)) {
            // End-of-generation
            // 終了トークン
            break;
        }

        char buf[256] = {};
        int n = llama_token_to_piece(m_model, new_token_id, buf, sizeof(buf), 0, true);
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
        batch = llama_batch_get_one(&new_token_id, 1);

        // Cut off if too long
        // レスポンスが長すぎる場合は打ち切り
        ++generated_token_count;
        if (generated_token_count > max_reply_tokens) {
            if (piece.find('\n') != std::string::npos) {
                qDebug() << "Cutting off at newline.";
                break;
            } else if (generated_token_count > max_reply_tokens + extra_cutoff_tokens) {
                qDebug() << "Cutting off after extra tokens.";
                break;
            }
        }

        // For immediate partial updates, we can process events
        QCoreApplication::processEvents();
    }

    // Update prev_len
    // 次の呼び出しに備えて prev_len を更新
    prev_len = llama_chat_apply_template(
        m_model,
        nullptr,
        messages_for_llama.data(),
        messages_for_llama.size(),
        false,
        nullptr,
        0
        );
    if (prev_len < 0) {
        fprintf(stderr, "Failed to apply chat template.\n");
    }

    // Emit final result
    // 最終的な結果を通知
    emit generationFinished(QString::fromStdString(response));
}

// to_llama_messages(...): convert from QList to std::vector
// to_llama_messages(...): QList -> std::vector 変換
std::vector<llama_chat_message> LlamaResponseGenerator::to_llama_messages(const QList<LlamaChatMessage> &user_messages)
{
    std::vector<llama_chat_message> llama_messages;
    llama_messages.reserve(user_messages.size());

    for (const auto &um : user_messages) {
        llama_chat_message lm;
        lm.role    = strdup(um.role().toUtf8().constData());
        lm.content = strdup(um.content().toUtf8().constData());
        llama_messages.push_back(lm);
    }
    return llama_messages;
}

// do_engine_init(): loads model/context in background
// do_engine_init(): バックグラウンドでモデル/コンテキストをロード
void LlamaResponseGenerator::do_engine_init()
{
    ggml_backend_load_all();

    m_model_params = llama_model_default_params();
    m_model_params.n_gpu_layers = m_n_gl;

    m_model = llama_load_model_from_file(m_model_path.c_str(), m_model_params);
    if (!m_model) {
        fprintf(stderr, "Error: unable to load model.\n");
        return;
    }

    m_ctx_params = llama_context_default_params();
    m_ctx_params.n_ctx   = m_n_ctx;
    m_ctx_params.n_batch = m_n_ctx;

    m_ctx = llama_new_context_with_model(m_model, m_ctx_params);
    if (!m_ctx) {
        fprintf(stderr, "Error: failed to create llama_context.\n");
        return;
    }

    // Initialize sampler
    // サンプラーの初期化
    m_sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(m_sampler, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(m_sampler, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(m_sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    // Notify remote side that init succeeded
    // リモート側に初期化成功を通知
    setRemoteInitialized(true);
    qDebug() << "Engine initialization complete.";
    qDebug() << "m_remoteInitialized =" << remoteInitialized();
}

// Default model path (CMake)
const std::string LlamaResponseGenerator::m_model_path {
#ifdef LLAMA_MODEL_FILE
    LLAMA_MODEL_FILE
#else
#error "LLAMA_MODEL_FILE is not defined. Please define it via target_compile_definitions() in CMake."
#endif
};
