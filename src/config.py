from pathlib import Path

root = Path(__file__).parent.parent

DATA_DIR = root / "data"
LOG_DIR = root / "logs"

ACTIVE = "active"
LAPSING = "lapsing"
LOST = "lost"

TRANSACTIONS_DATASET = DATA_DIR / "transactions.parquet"
CUSTOMER_DATA_DATASET = DATA_DIR / "customer_data.parquet"
BTYD_FEATURES_DATASET = DATA_DIR / "btyd_features.parquet"


TRANSACTION_QUERY = """
    SELECT customer_id as id, 
    DATETIME(transaction_completed_datetime) as txn_date,
    coalesce(inbound_payment_value - outbound_payment_value, 0) as value
    FROM `mpb-data-science-dev-ab-602d.dsci_daw.STV` 
    WHERE transaction_completed_datetime is not null
    """

CUSTOMER_DATA_QUERY = """

WITH scv AS (

    SELECT customer_id,
          market AS current_market,
          email_marketable AS is_email_marketable,
          DATE(first_transaction_completed_datetime) AS acquisition_date,
          current_customer_type,
          current_customer_tier,
          customer_three_year_gmv_lttv_gbp,
          customer_three_year_sell_value_lttv_gbp,
          buy_rfm_score,
          sell_rfm_score,
          first_channel AS acquisition_channel,
          first_sub_channel AS acquisition_sub_channel
    FROM `mpb-platform-prod-f68e.reference_db.SCV`
    WHERE first_transaction_completed_datetime IS NOT NULL
),

stv AS (

    SELECT DISTINCT customer_id,
           stv_updated_datetime,
           MIN(first_buy_date) AS first_buy_date,
           MIN(first_sell_date) AS first_sell_date,
           MAX(CASE WHEN transaction_order = 1 THEN transaction_type END) AS first_transaction_type,
           MAX(CASE WHEN transaction_order = 2 THEN DATE_DIFF(DATE(transaction_completed_datetime),first_transaction_date,DAY) END) AS days_to_second_transaction,
           MAX(CASE WHEN DATE(transaction_completed_datetime) BETWEEN DATE_ADD(CURRENT_DATE(),INTERVAL - 12 MONTH) AND DATE_ADD(CURRENT_DATE(),INTERVAL -1 DAY) THEN 1 ELSE 0 END) AS is_active_customer,
           MAX(CASE WHEN DATE(invoice_datetime) BETWEEN DATE_ADD(CURRENT_DATE(),INTERVAL - 12 MONTH) AND DATE_ADD(CURRENT_DATE(), INTERVAL-1 DAY) THEN 1 ELSE 0 END) AS is_active_buyer,
           MAX(CASE WHEN DATE(sell_datetime) BETWEEN DATE_ADD(CURRENT_DATE(),INTERVAL - 12 MONTH) AND DATE_ADD(CURRENT_DATE(), INTERVAL-1 DAY) THEN 1 ELSE 0 END) AS is_active_seller,
           IF(SUM(CASE WHEN DATE(transaction_completed_datetime) BETWEEN DATE_ADD(CURRENT_DATE(),INTERVAL - 24 MONTH) AND DATE_ADD(CURRENT_DATE(),INTERVAL -1 DAY) THEN 1 ELSE 0 END)>=3,1,0) AS is_habitual_customer,
           IF(SUM(CASE WHEN DATE(invoice_datetime) BETWEEN DATE_ADD(CURRENT_DATE(),INTERVAL - 24 MONTH) AND DATE_ADD(CURRENT_DATE(),INTERVAL -1 DAY) THEN 1 ELSE 0 END)>=3,1,0) AS is_habitual_buyer,
           IF(SUM(CASE WHEN DATE(sell_datetime) BETWEEN DATE_ADD(CURRENT_DATE(),INTERVAL - 24 MONTH) AND DATE_ADD(CURRENT_DATE(),INTERVAL -1 DAY) THEN 1 ELSE 0 END)>=3,1,0) AS is_habitual_seller,
           ABS(SUM(CASE WHEN market = "uk" THEN gmv
                        WHEN market = "us" THEN gmv/1.25
                   ELSE gmv/1.2
                   END)) AS customer_all_time_gmv_gbp,
           SUM(CASE WHEN market = "uk" THEN total_sold_sell_value
                        WHEN market = "us" THEN total_sold_sell_value/1.25
                   ELSE total_sold_sell_value/1.2
                   END) AS customer_all_time_sell_value_gbp,
           COUNT(DISTINCT transaction_id) AS total_completed_transactions,
           COUNT(DISTINCT CASE WHEN transaction_type IN ("BUYING","TRADING") THEN transaction_id END) AS total_buy_transactions,
           COUNT(DISTINCT CASE WHEN transaction_type IN ("SELLING","TRADING") THEN transaction_id END) AS total_sell_transactions,
           MAX(CASE WHEN reverse_order = 1 THEN
                    CASE WHEN transaction_type IN ("BUYING","TRADING") THEN txn_shipment_county_state 
                    ELSE txn_collection_county_state END
               ELSE NULL END) AS current_customer_region_state,
           MAX(CASE WHEN reverse_order = 1 THEN
                    CASE WHEN transaction_type IN ("BUYING","TRADING") THEN txn_shipment_country
                    ELSE txn_collection_country END
               ELSE NULL END) AS current_customer_country,
           MAX(transaction_completed_datetime) AS latest_transaction_date
    FROM 
        (SELECT *,
                DATE(MIN(transaction_completed_datetime)OVER(PARTITION BY customer_id)) AS first_transaction_date,
                DATE(MIN(CASE WHEN transaction_type IN ("BUYING","TRADING") THEN transaction_completed_datetime END)OVER(PARTITION BY customer_id)) AS first_buy_date,
                DATE(MIN(CASE WHEN transaction_type IN ("SELLING","TRADING") THEN transaction_completed_datetime END)OVER(PARTITION BY customer_id)) AS first_sell_date,
                ROW_NUMBER()OVER(PARTITION BY customer_id ORDER BY transaction_completed_datetime,transaction_reference) AS transaction_order,
                ROW_NUMBER()OVER(PARTITION BY customer_id ORDER BY transaction_completed_datetime DESC, transaction_reference DESC) AS reverse_order
        FROM `mpb-platform-prod-f68e.reference_db.STV`
        WHERE transaction_completed_datetime IS NOT NULL)
    GROUP BY customer_id, stv_updated_datetime

),

platform AS (

    SELECT DISTINCT cid.customer_id,
           COUNT(DISTINCT usv.fs_market_session_id) AS sessions,
           COUNT(DISTINCT CASE WHEN session_channel = "Organic" THEN fs_market_session_id END)/COUNT(DISTINCT fs_market_session_id) AS sessions_perc_organic,
           COUNT(DISTINCT DATE(usv.session_start_datetime)) AS days_active,
           COUNT(DISTINCT DATE(DATE_TRUNC(usv.session_start_datetime, MONTH))) AS months_active,
           COUNT(DISTINCT CASE WHEN session_device = "Mobile" THEN fs_market_session_id END)/COUNT(DISTINCT fs_market_session_id) AS sessions_perc_mobile,
           SUM(session_contains_sell_form_view) AS sessions_sell_form,
           SUM(session_contains_model_page_view) AS sessions_model_page_view,
           SUM(session_total_product_page_views) AS total_product_page_views,
           DATE_DIFF(MAX(CURRENT_DATE()),DATE(MAX(session_start_datetime)),DAY) AS session_recency,
           DATE_DIFF(MAX(CURRENT_DATE()),DATE(MAX(CASE WHEN session_contains_sell_form_view = 1 THEN session_start_datetime END)),DAY) AS sell_form_recency,
           DATE_DIFF(MAX(CURRENT_DATE()),DATE(MAX(CASE WHEN session_total_product_page_views > 0 THEN session_start_datetime END)),DAY) AS product_page_recency,
           SUM(session_active_duration_milliseconds) AS total_active_browsing_time_milliseconds
    FROM `mpb-platform-prod-f68e.fullstory.fs_user_id_customer_id_lookup` AS cid
        INNER JOIN 
          (SELECT fs_user_id,
                  fs_market_session_id,
                  session_channel,
                  session_start_datetime,
                  session_active_duration_milliseconds,
                  session_device,
                  session_contains_sell_form_view,
                  session_total_product_page_views,
                  session_contains_model_page_view
          FROM `mpb-platform-prod-f68e.fullstory.fs_unique_session_view`
          WHERE DATE(session_start_datetime) >= DATE_ADD(DATE_ADD(CURRENT_DATE(), INTERVAL -2 DAY), INTERVAL - 12 MONTH)) AS usv
        ON cid.fs_user_id = usv.fs_user_id
    GROUP BY cid.customer_id

),

product_in AS (

    SELECT DISTINCT customer_id,
           SUM(items) AS total_items_sold,
           SUM(items)/SUM(txns) AS avg_items_sold_per_txn,
           SUM(CASE WHEN CONTAINS_SUBSTR(UPPER(primary_category),"MIRRORLESS") THEN items ELSE 0 END) AS mirrorless_items_sold,
           SUM(CASE WHEN CONTAINS_SUBSTR(UPPER(primary_category),"DSLR") THEN items ELSE 0 END) AS dslr_items_sold,
           SUM(CASE WHEN CONTAINS_SUBSTR(UPPER(primary_category),"COMPACT") THEN items ELSE 0  END) AS compact_items_sold,
           SUM(CASE WHEN family = "Camera" THEN items ELSE 0 END) AS cameras_sold,
           SUM(CASE WHEN family = "Lens" THEN items ELSE 0 END) AS lenses_sold,
           SUM(CASE WHEN family = "Cine" THEN items ELSE 0 END) AS cine_kit_sold,
           MAX(CASE WHEN order_items = 1 THEN brand END) AS most_sold_brand,
           MAX(CASE WHEN order_txns = 1 THEN brand END) AS most_frequently_sold_brand,
           MAX(CASE WHEN order_items = 1 THEN primary_category END) AS most_sold_category
    FROM
        (SELECT *,
               ROW_NUMBER()OVER(PARTITION BY customer_id ORDER BY items DESC, value DESC) AS order_items,
               ROW_NUMBER()OVER(PARTITION BY customer_id ORDER BY txns DESC, items DESC, value DESC) AS order_txns
        FROM
            (SELECT DISTINCT customer_id,
                brand,
                family,
                primary_category,
                COUNT(DISTINCT line_item_id) AS items,
                COUNT(DISTINCT transaction_id) AS txns,
                SUM(value) AS value
            FROM
                (SELECT customer_id,
                    transaction_id,
                    line_item_id,
                    transaction_product_model_id,
                    INITCAP(brand) AS brand,
                    family,
                    INITCAP(primary_category) AS primary_category,
                    latest_product_quoted_value-total_adjustment_value AS value
                FROM `mpb-platform-prod-f68e.reference_db.line_item_view` AS liv
                    LEFT JOIN `mpb-platform-prod-f68e.reference_db.SMV` AS smv
                    ON liv.transaction_product_model_id = smv.model_id
                WHERE line_item_type = "GOODS IN"
                AND line_item_state = "CONFIRMED"
                AND transaction_variant IN ("SELLING","TRADING")
                AND product_sell_date IS NOT NULL)
            GROUP BY customer_id, brand, family,primary_category
            )
        )
    GROUP BY customer_id
),


product_out AS (

    SELECT DISTINCT customer_id,
           SUM(items) AS total_items_bought,
           SUM(items)/SUM(txns) AS avg_items_bought_per_txn,
           SUM(CASE WHEN CONTAINS_SUBSTR(UPPER(primary_category),"MIRRORLESS") THEN items ELSE 0  END) AS mirrorless_items_bought,
           SUM(CASE WHEN CONTAINS_SUBSTR(UPPER(primary_category),"DSLR") THEN items ELSE 0 END) AS dslr_items_bought,
           SUM(CASE WHEN CONTAINS_SUBSTR(UPPER(primary_category),"COMPACT") THEN items ELSE 0 END) AS compact_items_bought,
           SUM(CASE WHEN family = "Camera" THEN items ELSE 0 END) AS cameras_bought,
           SUM(CASE WHEN family = "Lens" THEN items ELSE 0 END) AS lenses_bought,
           SUM(CASE WHEN family = "Cine" THEN items ELSE 0 END) AS cine_kit_bought,
           MAX(CASE WHEN order_items = 1 THEN brand END) AS most_bought_brand,
           MAX(CASE WHEN order_txns = 1 THEN brand END) AS most_frequently_bought_brand,
           MAX(CASE WHEN order_items = 1 THEN primary_category END) AS most_bought_category
    FROM
        (SELECT *,
               ROW_NUMBER()OVER(PARTITION BY customer_id ORDER BY items DESC, value DESC) AS order_items,
               ROW_NUMBER()OVER(PARTITION BY customer_id ORDER BY txns DESC, items DESC, value DESC) AS order_txns
        FROM
            (SELECT DISTINCT customer_id,
                brand,
                family,
                primary_category,
                COUNT(DISTINCT line_item_id) AS items,
                COUNT(DISTINCT transaction_id) AS txns,
                SUM(value) AS value
            FROM
                (SELECT customer_id,
                    transaction_id,
                    line_item_id,
                    transaction_product_model_id,
                    INITCAP(brand) AS brand,
                    family,
                    INITCAP(primary_category) AS primary_category,
                    latest_product_quoted_value-total_adjustment_value AS value
                FROM `mpb-platform-prod-f68e.reference_db.line_item_view` AS liv
                    LEFT JOIN `mpb-platform-prod-f68e.reference_db.SMV` AS smv
                    ON liv.transaction_product_model_id = smv.model_id
                WHERE line_item_type = "GOODS OUT"
                AND line_item_state = "CONFIRMED"
                AND transaction_invoice_datetime IS NOT NULL
                AND transaction_variant IN ("BUYING","TRADING"))
            GROUP BY customer_id, brand, family,primary_category
            )
        )
    GROUP BY customer_id
)

SELECT 
    scv.*,
    stv.first_buy_date,
    stv.first_sell_date,
    stv.first_transaction_type,
    stv.days_to_second_transaction,
    stv.is_active_customer,
    stv.is_active_buyer,
    stv.is_active_seller,
    stv.is_habitual_customer,
    stv.is_habitual_buyer,
    stv.is_habitual_seller,
    stv.customer_all_time_gmv_gbp,
    stv.customer_all_time_sell_value_gbp,
    stv.total_completed_transactions,
    stv.total_buy_transactions,
    stv.total_sell_transactions,
    stv.current_customer_region_state,
    stv.current_customer_country,
    stv.latest_transaction_date,
    platform.sessions,
    platform.sessions_perc_organic,
    platform.days_active,
    platform.months_active,
    platform.sessions_perc_mobile,
    platform.sessions_sell_form,
    platform.sessions_model_page_view,
    platform.total_product_page_views,
    platform.session_recency,
    platform.sell_form_recency,
    platform.product_page_recency,
    platform.total_active_browsing_time_milliseconds,
    product_in.total_items_sold,
    product_in.avg_items_sold_per_txn,
    product_in.mirrorless_items_sold,
    product_in.dslr_items_sold,
    product_in.compact_items_sold,
    product_in.cameras_sold,
    product_in.lenses_sold,
    product_in.cine_kit_sold,
    product_in.most_sold_brand,
    product_in.most_frequently_sold_brand,
    product_in.most_sold_category,
    product_out.total_items_bought,
    product_out.avg_items_bought_per_txn,
    product_out.mirrorless_items_bought,
    product_out.dslr_items_bought,
    product_out.compact_items_bought,
    product_out.cameras_bought,
    product_out.lenses_bought,
    product_out.cine_kit_bought,
    product_out.most_bought_brand,
    product_out.most_frequently_bought_brand,
    product_out.most_bought_category
FROM scv
    LEFT JOIN stv ON scv.customer_id = stv.customer_id
    LEFT JOIN platform ON scv.customer_id = platform.customer_id
    LEFT JOIN product_in ON scv.customer_id = product_in.customer_id
    LEFT JOIN product_out ON scv.customer_id = product_out.customer_id

"""
