# Shared loading, filtering and inference helpers for the cross-condition notebooks.

if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(tidyverse, arrow, here)

PROJECT_ROOT <- here()

# =============================================================================
# Conditions
# =============================================================================

ALL_CONDITIONS <- c("human-ai", "human-human", "ai-ai", "ai-ai-cross")

CONDITION_LABELS <- c(
  "human-ai"    = "Human-AI",
  "human-human" = "Human-Human",
  "ai-ai"       = "AI-AI",
  "ai-ai-cross" = "AI-AI-Cross"
)

# The paper's planned contrasts are defined over these three. Set
# CONDITIONS <- PUBLISHED_CONDITIONS in a notebook to drop ai-ai-cross.
PUBLISHED_CONDITIONS <- c("human-ai", "human-human", "ai-ai")

condition_colors <- c(
  "Human-AI"    = "#1f77b4",
  "Human-Human" = "#2ca02c",
  "AI-AI"       = "#ff7f0e",
  "AI-AI-Cross" = "#8c564b"
)

# Stories excluded from the valence and novelty analyses (provider errors and
# broken sessions).
EXCLUDED_CONVERSATIONS <- c(
  "conv_ed575a06c11d42358e3eeb7826d2f959",
  "conv_a63a08273d0a4704a7638e4cd6850225",
  "conv_0bb56093-3033-4615-bb70-ebfa4135589a",
  "conv_0f18b30f-7d4b-4681-b98e-a0ff4f2b5256",
  "conv_72218cb5-e59c-4c93-a4b9-a057fe5dad80"
)

# =============================================================================
# Cleaning helpers
# =============================================================================

clean_conversation_id <- function(x) {
  if (is.list(x)) {
    x <- purrr::map_chr(x, ~ paste(unlist(.x), collapse = ""))
  }
  as.character(x) %>%
    str_replace_all("^\\[|\\]$", "") %>%
    str_replace_all("^['\"]|['\"]$", "")
}

coerce_numeric_cols <- function(df, cols = c("turn", "interaction_count", "analysis_turn")) {
  present <- intersect(cols, names(df))
  df %>% mutate(across(all_of(present), ~ suppressWarnings(as.numeric(.x))))
}

coerce_text_reason_cols <- function(df) {
  reason_cols <- names(df)[str_detect(names(df), "(^|_)qc_reasons$|_reasons$")]
  df %>% mutate(across(all_of(reason_cols), ~ na_if(as.character(.x), "")))
}

normalize_condition_id <- function(x) {
  as.character(x) %>% str_to_lower() %>% str_replace_all("_", "-")
}

# =============================================================================
# Provider errors
# =============================================================================
# Failed model calls are excluded post hoc. Turn-level analyses drop only the
# failed exchange; story-level analyses drop the whole story.

provider_error_patterns <- c(
  "Network error while generating a response",
  "Claude adapter: no text returned from provider"
)

provider_error_exchanges <- tibble::tribble(
  ~condition, ~conversation_id, ~turn,
  "human-ai", "conv_0f18b30f-7d4b-4681-b98e-a0ff4f2b5256", 6,
  "human-ai", "conv_72218cb5-e59c-4c93-a4b9-a057fe5dad80", 7,
  "human-ai", "conv_7c23347e-6172-4c84-9fd0-45ef34290bd5", 3
)

provider_error_stories <- tibble::tribble(
  ~condition, ~conversation_id,
  "ai-ai", "conv_a79338efb1384551affc0d7597822b0f"
)

is_provider_error_text <- function(x) {
  replace_na(str_detect(
    as.character(x),
    regex(paste(provider_error_patterns, collapse = "|"), ignore_case = TRUE)
  ), FALSE)
}

row_condition_ids <- function(df) {
  if ("condition" %in% names(df)) normalize_condition_id(df$condition) else rep(NA_character_, nrow(df))
}

drop_provider_error_exchanges <- function(df) {
  if (!"conversation_id" %in% names(df) || nrow(df) == 0) {
    return(df)
  }
  df <- df %>% mutate(conversation_id = clean_conversation_id(conversation_id))
  keys <- tibble(.row = seq_len(nrow(df)), condition = row_condition_ids(df), conversation_id = df$conversation_id)

  drop <- rep(FALSE, nrow(df))
  text_cols <- intersect(
    c("author_1", "author_2", "text", "full_text", "story_text", "author_1_text", "author_2_text"),
    names(df)
  )
  for (col in text_cols) {
    drop <- drop | is_provider_error_text(df[[col]])
  }

  story_rows <- keys %>% inner_join(provider_error_stories, by = c("condition", "conversation_id")) %>% pull(.row)
  drop[story_rows] <- TRUE

  turn_cols <- intersect(c("turn", "interaction_count"), names(df))
  if (length(turn_cols) > 0) {
    turn <- Reduce(coalesce, lapply(turn_cols, function(col) suppressWarnings(as.numeric(df[[col]]))))
    exchange_rows <- keys %>%
      mutate(turn = turn) %>%
      inner_join(provider_error_exchanges, by = c("condition", "conversation_id", "turn")) %>%
      pull(.row)
    drop[exchange_rows] <- TRUE
  }

  df %>% filter(!drop)
}

drop_provider_error_stories <- function(df) {
  if (!"conversation_id" %in% names(df) || nrow(df) == 0) {
    return(df)
  }
  bad <- bind_rows(provider_error_exchanges %>% select(condition, conversation_id), provider_error_stories) %>%
    distinct()
  df <- df %>% mutate(conversation_id = clean_conversation_id(conversation_id))
  rows <- tibble(.row = seq_len(nrow(df)), condition = row_condition_ids(df), conversation_id = df$conversation_id) %>%
    inner_join(bad, by = c("condition", "conversation_id")) %>%
    pull(.row)
  if (length(rows) == 0) df else df[-rows, , drop = FALSE]
}

# =============================================================================
# Metadata
# =============================================================================

load_interaction_metadata <- function(condition) {
  path <- here("data", condition, "interim", "interaction_level_stories_filtered.csv")
  if (!file.exists(path)) {
    warning(paste("Interaction metadata not found:", path))
    return(NULL)
  }
  read_csv(path, show_col_types = FALSE) %>%
    mutate(timestamp = if ("timestamp" %in% names(.)) as.character(timestamp) else NULL) %>%
    mutate(conversation_id = clean_conversation_id(conversation_id)) %>%
    coerce_numeric_cols()
}

# Joins the interim exchange metadata onto a processed table: per turn when the
# table has turn keys, otherwise per story. Exchange-alignment columns prefer
# the interim values; all other columns keep the processed values.
attach_interaction_metadata <- function(df, condition) {
  if (!"conversation_id" %in% names(df)) {
    return(df)
  }
  meta <- load_interaction_metadata(condition)
  df <- df %>% mutate(conversation_id = clean_conversation_id(conversation_id))
  if (is.null(meta)) {
    return(df)
  }
  df <- coerce_numeric_cols(df)

  keys <- intersect(c("conversation_id", "turn", "interaction_count"), intersect(names(df), names(meta)))
  if (length(keys) >= 2) {
    meta <- meta %>% distinct(!!!rlang::syms(keys), .keep_all = TRUE)
  } else {
    story_cols <- intersect(
      c("conversation_id", "starter", "starter_side", "starter_type",
        "author_1", "author_2", "author_1_type", "author_2_type",
        "respondent_id", "respondent_id_u1", "respondent_id_u2",
        "llm_type", "model_id",
        "model_starter", "model_responder", "pair_id", "dyad_id", "is_self_pair"),
      names(meta)
    )
    meta <- meta %>% select(all_of(story_cols)) %>% distinct(conversation_id, .keep_all = TRUE)
    keys <- "conversation_id"
  }

  joined <- df %>% left_join(meta, by = keys, suffix = c("", ".meta"))
  prefer_interim <- c("analysis_turn", "complete_exchange", "starter", "starter_side",
                      "starter_type", "author_1_type", "author_2_type")

  meta_cols <- names(joined)[str_detect(names(joined), "\\.meta$")]
  for (meta_col in meta_cols) {
    base_col <- str_remove(meta_col, "\\.meta$")
    cast <- tryCatch(vctrs::vec_cast(joined[[meta_col]], joined[[base_col]]), error = function(e) NULL)
    if (!is.null(cast)) {
      joined[[base_col]] <- if (base_col %in% prefer_interim) {
        coalesce(cast, joined[[base_col]])
      } else {
        coalesce(joined[[base_col]], cast)
      }
    }
  }
  joined %>% select(-all_of(meta_cols))
}

ensure_role_metadata <- function(df) {
  df <- df %>% mutate(conversation_id = clean_conversation_id(conversation_id))
  if (!"starter_side" %in% names(df) && "starter" %in% names(df)) {
    df <- df %>% mutate(starter_side = starter)
  }
  char_cols <- intersect(c("speaker_slot", "speaker_type", "partner_type", "starter_side", "starter_type"), names(df))
  df <- df %>% mutate(across(all_of(char_cols), ~ na_if(as.character(.x), "")))
  if ("analysis_turn" %in% names(df)) {
    df <- df %>% mutate(analysis_turn = suppressWarnings(as.numeric(analysis_turn)))
  }
  if ("complete_exchange" %in% names(df)) {
    df <- df %>% mutate(complete_exchange = as.logical(complete_exchange))
  }
  df
}

add_turn_index <- function(df) {
  if ("analysis_turn" %in% names(df) && any(!is.na(df$analysis_turn))) {
    return(df %>% mutate(turn_index = analysis_turn))
  }
  if ("turn" %in% names(df)) {
    return(df %>% mutate(turn_index = suppressWarnings(as.numeric(turn))))
  }
  if ("interaction_count" %in% names(df)) {
    return(df %>% mutate(turn_index = suppressWarnings(as.numeric(interaction_count))))
  }
  df
}

# Human / AI in Human-AI; starter side otherwise.
derive_condition_role <- function(df) {
  df %>%
    ensure_role_metadata() %>%
    mutate(
      condition_role = case_when(
        condition == "human-ai" & !is.na(speaker_type) & speaker_type == "human" ~ "Human",
        condition == "human-ai" & !is.na(speaker_type) & speaker_type == "ai" ~ "AI",
        !is.na(speaker_is_starter) & speaker_is_starter ~ "Starter-side",
        !is.na(speaker_is_starter) & !speaker_is_starter ~ "Non-starter-side",
        speaker_slot == "author_1" ~ "Slot 1",
        speaker_slot == "author_2" ~ "Slot 2",
        TRUE ~ NA_character_
      ),
      slot_label = case_when(
        speaker_slot == "author_1" ~ "Slot 1",
        speaker_slot == "author_2" ~ "Slot 2",
        TRUE ~ NA_character_
      )
    )
}

# Valence of the chronologically first and second turn of each exchange.
add_chronological_valence <- function(df) {
  df %>%
    mutate(
      chronological_first_slot = case_when(
        normalize_condition_id(condition) == "human-ai" ~ "author_1",
        starter_side == "author_2" ~ "author_2",
        TRUE ~ "author_1"
      ),
      first_valence = if_else(chronological_first_slot == "author_1", author_1_valence, author_2_valence),
      second_valence = if_else(chronological_first_slot == "author_1", author_2_valence, author_1_valence)
    )
}

# Complete exchanges in analysis_turn 1-9, without provider errors.
filter_complete_exchange_window <- function(df, min_analysis_turn = 1, max_analysis_turn = 9) {
  df <- drop_provider_error_exchanges(df)
  if ("complete_exchange" %in% names(df)) {
    df <- df %>% filter(is.na(complete_exchange) | complete_exchange)
  }
  turn_col <- if ("analysis_turn" %in% names(df) && any(!is.na(df$analysis_turn))) {
    "analysis_turn"
  } else {
    intersect(c("turn", "interaction_count"), names(df))[1]
  }
  if (!is.na(turn_col)) {
    df <- df %>% filter(between(suppressWarnings(as.numeric(.data[[turn_col]])), min_analysis_turn, max_analysis_turn))
  }
  add_turn_index(df)
}

prepare_condition_factor <- function(df, reference = "human-ai") {
  levels <- c(reference, setdiff(ALL_CONDITIONS, reference))
  levels <- intersect(levels, unique(as.character(df$condition)))
  df %>% mutate(condition = factor(condition, levels = levels, labels = CONDITION_LABELS[levels]))
}

# =============================================================================
# Loading
# =============================================================================

load_condition_data <- function(condition, filename, format = "parquet") {
  path <- here("data", condition, "processed", filename)
  if (!file.exists(path)) {
    warning(paste("File not found:", path))
    return(NULL)
  }
  df <- if (format == "parquet") read_parquet(path) else read_csv(path, show_col_types = FALSE)
  if ("timestamp" %in% names(df)) {
    df <- df %>% mutate(timestamp = as.character(timestamp))
  }
  df %>%
    mutate(
      condition = condition,
      conversation_id = if ("conversation_id" %in% names(.)) clean_conversation_id(conversation_id) else NULL
    ) %>%
    coerce_text_reason_cols() %>%
    coerce_numeric_cols()
}

load_conditions <- function(conditions, filename, format = "parquet", prepare = identity) {
  dfs <- purrr::map(conditions, function(cond) {
    df <- suppressWarnings(load_condition_data(cond, filename, format))
    if (is.null(df)) {
      return(NULL)
    }
    df %>% attach_interaction_metadata(cond) %>% ensure_role_metadata() %>% prepare()
  }) %>% compact()
  if (length(dfs) == 0) stop("No data found for ", filename)
  bind_rows(dfs)
}

# Published valence: each turn embedded on its own and projected onto the
# sentiment concept vector (author_*_sentiment_projection).
add_valence <- function(df) {
  for (slot in c("author_1", "author_2")) {
    source <- paste0(slot, "_sentiment_projection")
    if (source %in% names(df)) df[[paste0(slot, "_valence")]] <- df[[source]]
  }
  df
}

load_valence_data <- function(conditions = ALL_CONDITIONS) {
  dfs <- purrr::map(conditions, function(cond) {
    df <- suppressWarnings(load_condition_data(cond, "dyadic_sentiment_scores.parquet"))
    if (is.null(df)) {
      return(NULL)
    }
    df %>% add_valence() %>% attach_interaction_metadata(cond) %>% ensure_role_metadata() %>% add_turn_index()
  }) %>% compact()
  if (length(dfs) == 0) stop("No valence data found")
  bind_rows(dfs)
}

load_novelty_data <- function(conditions = ALL_CONDITIONS) {
  load_conditions(conditions, "novelty_scores.csv", format = "csv", prepare = add_turn_index)
}

load_exploration_data <- function(conditions = ALL_CONDITIONS) {
  load_conditions(conditions, "semantic_exploration_binned.parquet",
                  prepare = function(df) df %>% derive_condition_role() %>% add_turn_index() %>% drop_provider_error_stories())
}

# =============================================================================
# Plotting
# =============================================================================

theme_comparison <- function(base_size = 14) {
  theme_minimal(base_size = base_size) +
    theme(
      legend.position = "bottom",
      panel.grid.minor = element_blank(),
      strip.text = element_text(face = "bold"),
      plot.title = element_text(hjust = 0.5, face = "bold"),
      plot.subtitle = element_text(hjust = 0.5)
    )
}

# =============================================================================
# Inference on per-story asymmetry values (one row per story)
# =============================================================================

bootstrap_condition_means <- function(df, value_col, R = 10000, ci = 0.95, seed = 42) {
  set.seed(seed)
  alpha <- (1 - ci) / 2
  df %>%
    filter(!is.na(.data[[value_col]])) %>%
    group_by(condition) %>%
    group_modify(function(g, key) {
      x <- g[[value_col]]
      n <- length(x)
      if (n < 2) {
        return(tibble(n = n, mean = mean(x), ci_lower = NA_real_, ci_upper = NA_real_))
      }
      boots <- replicate(R, mean(sample(x, n, replace = TRUE)))
      tibble(
        n = n,
        mean = mean(x),
        ci_lower = unname(quantile(boots, alpha)),
        ci_upper = unname(quantile(boots, 1 - alpha))
      )
    }) %>%
    ungroup()
}

# Cliff's delta for every pair of conditions.
effect_sizes_pairwise <- function(df, value_col) {
  conds <- levels(df$condition)
  if (is.null(conds)) conds <- sort(unique(as.character(df$condition)))

  cliff_delta <- function(x, y) {
    x <- x[!is.na(x)]
    y <- y[!is.na(y)]
    if (length(x) == 0 || length(y) == 0) return(NA_real_)
    (sum(outer(x, y, ">")) - sum(outer(x, y, "<"))) / (length(x) * length(y))
  }

  purrr::map_dfr(utils::combn(conds, 2, simplify = FALSE), function(p) {
    a <- df[[value_col]][as.character(df$condition) == p[[1]]]
    b <- df[[value_col]][as.character(df$condition) == p[[2]]]
    tibble(
      contrast = paste(p[[1]], "vs", p[[2]]),
      n_a = sum(!is.na(a)),
      n_b = sum(!is.na(b)),
      cliff_delta = cliff_delta(a, b),
      magnitude = case_when(
        is.na(cliff_delta) ~ NA_character_,
        abs(cliff_delta) < 0.147 ~ "negligible",
        abs(cliff_delta) < 0.330 ~ "small",
        abs(cliff_delta) < 0.474 ~ "medium",
        TRUE ~ "large"
      )
    )
  })
}

# Contrast weights for an emmeans grid, named by condition label; unnamed
# conditions get weight 0, so a contrast keeps its meaning when ai-ai-cross is loaded.
condition_weights <- function(emm, weights) {
  levs <- as.character(emm@grid$condition)
  missing <- setdiff(names(weights), levs)
  if (length(missing) > 0) stop("condition_weights: not in the grid: ", paste(missing, collapse = ", "))
  w <- setNames(rep(0, length(levs)), levs)
  w[names(weights)] <- weights
  unname(w)
}

HA_VS_SAME_TYPE <- c("Human-AI" = 1, "Human-Human" = -0.5, "AI-AI" = -0.5)

# Welch-type test of HA vs (HH + AA) / 2. AI-AI-Cross rows are ignored.
planned_contrast_HA_vs_same <- function(df, value_col) {
  vals <- function(label) df[[value_col]][as.character(df$condition) == label]
  ha <- vals("Human-AI")
  hh <- vals("Human-Human")
  aa <- vals("AI-AI")

  m <- function(x) mean(x, na.rm = TRUE)
  vn <- function(x) var(x, na.rm = TRUE) / sum(!is.na(x))
  n <- function(x) sum(!is.na(x))

  est <- m(ha) - 0.5 * (m(hh) + m(aa))
  terms <- c(vn(ha), 0.25 * vn(hh), 0.25 * vn(aa))
  se <- sqrt(sum(terms))
  df_welch <- sum(terms)^2 / sum(terms^2 / (c(n(ha), n(hh), n(aa)) - 1))
  t_stat <- est / se

  tibble(
    contrast = "Human-AI vs (Human-Human + AI-AI)/2",
    estimate = est,
    se = se,
    t = t_stat,
    df = df_welch,
    p = 2 * pt(-abs(t_stat), df = df_welch),
    ci_lower = est - qt(0.975, df_welch) * se,
    ci_upper = est + qt(0.975, df_welch) * se
  )
}

# Permutation test of the one-way ANOVA F statistic over story-level condition labels.
permutation_test_A <- function(df, value_col, R = 10000, seed = 42) {
  set.seed(seed)
  d <- df[!is.na(df[[value_col]]), c("condition", value_col)]
  d$condition <- as.factor(d$condition)
  if (nlevels(d$condition) < 2 || nrow(d) < 4) {
    return(tibble(F_obs = NA_real_, p_perm = NA_real_, R = R))
  }
  f_stat <- function(values, groups) summary(aov(values ~ groups))[[1]][["F value"]][1]
  obs <- f_stat(d[[value_col]], d$condition)
  null_F <- replicate(R, f_stat(d[[value_col]], sample(d$condition)))
  tibble(F_obs = obs, p_perm = (sum(null_F >= obs, na.rm = TRUE) + 1) / (sum(!is.na(null_F)) + 1), R = R)
}

# Whether a signed per-story asymmetry points the same way across stories:
# |mean|/sd, share of the majority sign and a one-sample t-test per condition,
# plus a Brown-Forsythe test of equal variances across conditions.
directional_stability <- function(df, value_col) {
  d <- df[!is.na(df[[value_col]]), c("condition", value_col)]
  if (!nrow(d)) {
    return(list(by_condition = tibble(), variance_equality_p = NA_real_))
  }
  d$condition <- as.factor(d$condition)

  by_condition <- d %>%
    group_by(condition) %>%
    summarise(
      n = n(),
      mean = mean(.data[[value_col]]),
      sd = sd(.data[[value_col]]),
      stability = abs(mean(.data[[value_col]])) / sd(.data[[value_col]]),
      sign_consistency = max(mean(.data[[value_col]] > 0), mean(.data[[value_col]] < 0)),
      one_sample_t_p = tryCatch(t.test(.data[[value_col]])$p.value, error = function(e) NA_real_),
      .groups = "drop"
    )

  bf_p <- if (nlevels(d$condition) >= 2 && nrow(d) >= 6) {
    tryCatch({
      med <- ave(d[[value_col]], d$condition, FUN = median)
      summary(aov(abs(d[[value_col]] - med) ~ d$condition))[[1]][["Pr(>F)"]][1]
    }, error = function(e) NA_real_)
  } else {
    NA_real_
  }

  list(by_condition = by_condition, variance_equality_p = bf_p)
}
