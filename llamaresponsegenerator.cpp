#include "llamaresponsegenerator.h"
#include <QDebug>
#include <QtConcurrent>

// Constructor: stores model/context, starts doEngineInit() in background
// コンストラクタ: model/contextを保持し、バックグラウンドでdoEngineInit()を開始
LlamaResponseGenerator::LlamaResponseGenerator(QObject *parent,
                                               llama_model *model,
                                               llama_context *ctx)
    : LlamaResponseGeneratorSimpleSource(parent)
    , m_model(model)
    , m_ctx(ctx)
{
    QtConcurrent::run(&LlamaResponseGenerator::doEngineInit, this);
}

// Destructor: free sampler if allocated
// デストラクタ: samplerがあれば解放
LlamaResponseGenerator::~LlamaResponseGenerator()
{
    if (m_sampler) {
        llama_sampler_free(m_sampler);
        m_sampler = nullptr;
    }
}

// generate(...) from .rep: tokenize prompt, decode tokens, emit partial/final
// .repファイルのgenerate(...): プロンプトをトークナイズしてデコードし、途中/最終結果をemit
void LlamaResponseGenerator::generate(const QList<LlamaChatMessage>& messages)
{
    static std::vector<char> formatted(llama_n_ctx(m_ctx));
    static int prev_len {0};

    std::vector<llama_chat_message> messages_for_llama = toLlamaMessages(messages);

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
    // Resize if needed
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

    std::string promptStd(formatted.begin() + prev_len, formatted.begin() + new_len);
    std::string response;

    // Tokenize prompt text. Negative means ignore special tokens, return count
    // プロンプトをトークン化。負数で特別トークンを無視してトークン数取得
    const int n_prompt_tokens = -llama_tokenize(
        m_model,
        promptStd.c_str(),
        promptStd.size(),
        nullptr,
        0,
        true,  // is_prefix
        true   // is_bos
        );

    // Store tokens in vector
    // トークンをベクタに格納
    std::vector<llama_token> prompt_tokens(n_prompt_tokens);
    if (llama_tokenize(
            m_model,
            promptStd.c_str(),
            promptStd.size(),
            prompt_tokens.data(),
            prompt_tokens.size(),
            llama_get_kv_cache_used_cells(m_ctx) == 0,
            true) < 0) {
        emit generationError("failed to tokenize the prompt");
    }

    // Prepare batch for decoding
    // デコード用のバッチ作成
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(),
                                            prompt_tokens.size());
    llama_token new_token_id;

    static constexpr int max_reply_tokens {1024};
    static constexpr int extra_cutoff_tokens {32};
    int generated_token_count {0};

    // Decode tokens until end-of-generation
    // 終了トークンが出るまでトークンをデコード
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
            // End-of-generation token
            // 終了トークン
            break;
        }

        // Convert token to text
        // トークンを文字列に変換
        char buf[256] = {};
        int n = llama_token_to_piece(m_model, new_token_id, buf,
                                     sizeof(buf), 0, true);
        if (n < 0) {
            emit generationError("failed to convert token to piece");
            break;
        }

        std::string piece(buf, n);
        printf("%s", piece.c_str());
        fflush(stdout);

        // Append partial piece
        // 部分的な出力を蓄積
        response += piece;

        // Emit partial progress
        // 中間進捗を通知
        emit partialResponseReady(QString::fromStdString(response));

        // Prepare next batch
        // 次のバッチを作成
        batch = llama_batch_get_one(&new_token_id, 1);

        // Cut off if too long
        // 長すぎる場合に打ち切り
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
    }

    // Update prev_len
    // prev_lenを更新
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

    // Emit final response
    // 最終結果を通知
    emit generationFinished(QString::fromStdString(response));
}

// userMessages (QList<LlamaChatMessage>) → llamaMessages (std::vector<llama_chat_message>)
std::vector<llama_chat_message> LlamaResponseGenerator::toLlamaMessages(const QList<LlamaChatMessage> &userMessages)
{
    std::vector<llama_chat_message> llamaMessages;
    llamaMessages.reserve(userMessages.size());

    for (const auto &um : userMessages) {
        llama_chat_message lm;
        lm.role = strdup(um.role().toUtf8().constData());
        lm.content = strdup(um.content().toUtf8().constData());
        llamaMessages.push_back(lm);
    }

    return llamaMessages;
}

// Heavy init in separate thread: load model/context
// 別スレッドでの重い初期化: モデル/コンテキストをロード
void LlamaResponseGenerator::doEngineInit()
{
    ggml_backend_load_all();

    m_model_params = llama_model_default_params();
    m_model_params.n_gpu_layers = m_ngl;

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
    // サンプラーを初期化
    m_sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(m_sampler, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(m_sampler, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(m_sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    // Indicate to the remote replica that initialization succeeded
    // リモート複製側に、初期化完了を通知
    setRemoteInitialized(true);
    qDebug() << "Engine initialization complete.";
    qDebug() << "m_remoteInitialized = " << remoteInitialized();
}

// Default model path set via CMake
// CMakeで設定されたデフォルトモデルパス
const std::string LlamaResponseGenerator::m_model_path {
#ifdef LLAMA_MODEL_FILE
    LLAMA_MODEL_FILE
#else
#error "LLAMA_MODEL_FILE is not defined. Please define it via target_compile_definitions() in CMake."
#endif
};
