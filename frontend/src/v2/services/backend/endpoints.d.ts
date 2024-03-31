/**
 * This file was auto-generated by openapi-typescript.
 * Do not make direct changes to the file.
 */


export interface paths {
  "/query/v1/app/cards": {
    /** Get cards */
    get: operations["__query_v1_app_cards_get"];
  };
  "/query/v1/app/cards/thumbnail/{image_id}.webp": {
    /** Card thumbnail (144x144) */
    get: operations["__query_v1_app_cards_thumbnail__image_id__webp_get"];
  };
  "/query/v1/app/cards/download": {
    /** Download image */
    get: operations["__query_v1_app_cards_download_get"];
  };
  "/command/v1/images/upload-local": {
    /** Upload local images */
    post: operations["__command_v1_images_upload_local_post"];
  };
  "/command/v1/images/{image_id}/delete": {
    /** Delete image */
    delete: operations["__command_v1_images__image_id__delete_delete"];
  };
  "/command/v1/images/{image_id}/process/run": {
    /** Run process */
    post: operations["__command_v1_images__image_id__process_run_post"];
  };
  "/command/v1/images/{image_id}/process/retry": {
    /** Retry latest process */
    post: operations["__command_v1_images__image_id__process_retry_post"];
  };
  "/command/v1/images/{image_id}/process/stop": {
    /** Stop latest process */
    delete: operations["__command_v1_images__image_id__process_stop_delete"];
  };
}

export type webhooks = Record<string, never>;

export interface components {
  schemas: {
    /** Body___command_v1_images_upload_local_post */
    Body___command_v1_images_upload_local_post: {
      /** Files */
      files: string[];
    };
    /** CardMod */
    CardMod: {
      /** Thumbnail Src */
      thumbnail_src: string;
      /** Name */
      name: string;
      source: components["schemas"]["Dimension"];
      target: components["schemas"]["Dimension"] | null;
      /** Status */
      status: components["schemas"]["Runnable"] | components["schemas"]["Stoppable"] | components["schemas"]["Errored"] | components["schemas"]["Downloadable"];
      /**
       * Extension
       * @enum {string}
       */
      extension: "JPEG" | "PNG" | "WEBP";
      /** Preserve Ratio */
      preserve_ratio: boolean;
      /** Enable Ai */
      enable_ai: boolean;
      /** Image Id */
      image_id: number;
    };
    /** Dimension */
    Dimension: {
      /** Width */
      width: number;
      /** Height */
      height: number;
    };
    /** Downloadable */
    Downloadable: {
      /**
       * Type
       * @constant
       */
      type: "Downloadable";
      /** Image Src */
      image_src: string;
    };
    /** Errored */
    Errored: {
      /**
       * Type
       * @constant
       */
      type: "Errored";
      /** Duration */
      duration: number;
      /** Error */
      error: string;
    };
    /**
     * ExtensionVal
     * @enum {string}
     */
    ExtensionVal: "JPEG" | "PNG" | "WEBP";
    /** HTTPValidationError */
    HTTPValidationError: {
      /** Detail */
      detail?: components["schemas"]["ValidationError"][];
    };
    /** ImageSizeDto */
    ImageSizeDto: {
      /** Width */
      width: number;
      /** Height */
      height: number;
    };
    /** ProcessDto */
    ProcessDto: {
      extension: components["schemas"]["ExtensionVal"];
      target: components["schemas"]["ImageSizeDto"];
      /** Enable Ai */
      enable_ai: boolean;
    };
    /** Runnable */
    Runnable: {
      /**
       * Type
       * @constant
       */
      type: "Runnable";
    };
    /** Stoppable */
    Stoppable: {
      /**
       * Type
       * @constant
       */
      type: "Stoppable";
      /**
       * Started At
       * Format: date-time
       */
      started_at: string;
      /** Duration */
      duration: number;
    };
    /** ValidationError */
    ValidationError: {
      /** Location */
      loc: (string | number)[];
      /** Message */
      msg: string;
      /** Error Type */
      type: string;
    };
  };
  responses: never;
  parameters: never;
  requestBodies: never;
  headers: never;
  pathItems: never;
}

export type $defs = Record<string, never>;

export type external = Record<string, never>;

export interface operations {

  /** Get cards */
  __query_v1_app_cards_get: {
    responses: {
      /** @description Successful Response */
      200: {
        content: {
          "application/json": components["schemas"]["CardMod"][];
        };
      };
    };
  };
  /** Card thumbnail (144x144) */
  __query_v1_app_cards_thumbnail__image_id__webp_get: {
    parameters: {
      path: {
        image_id: number;
      };
    };
    responses: {
      /** @description Successful Response */
      200: {
        content: never;
      };
      /** @description Validation Error */
      422: {
        content: {
          "application/json": components["schemas"]["HTTPValidationError"];
        };
      };
    };
  };
  /** Download image */
  __query_v1_app_cards_download_get: {
    parameters: {
      query: {
        image_id: number;
      };
    };
    responses: {
      /** @description Successful Response */
      200: {
        content: never;
      };
      /** @description Validation Error */
      422: {
        content: {
          "application/json": components["schemas"]["HTTPValidationError"];
        };
      };
    };
  };
  /** Upload local images */
  __command_v1_images_upload_local_post: {
    requestBody: {
      content: {
        "multipart/form-data": components["schemas"]["Body___command_v1_images_upload_local_post"];
      };
    };
    responses: {
      /** @description Successful Response */
      204: {
        content: never;
      };
      /** @description Validation Error */
      422: {
        content: {
          "application/json": components["schemas"]["HTTPValidationError"];
        };
      };
    };
  };
  /** Delete image */
  __command_v1_images__image_id__delete_delete: {
    parameters: {
      path: {
        image_id: number;
      };
    };
    responses: {
      /** @description Successful Response */
      204: {
        content: never;
      };
      /** @description Validation Error */
      422: {
        content: {
          "application/json": components["schemas"]["HTTPValidationError"];
        };
      };
    };
  };
  /** Run process */
  __command_v1_images__image_id__process_run_post: {
    parameters: {
      path: {
        image_id: number;
      };
    };
    requestBody: {
      content: {
        "application/json": components["schemas"]["ProcessDto"];
      };
    };
    responses: {
      /** @description Successful Response */
      204: {
        content: never;
      };
      /** @description Validation Error */
      422: {
        content: {
          "application/json": components["schemas"]["HTTPValidationError"];
        };
      };
    };
  };
  /** Retry latest process */
  __command_v1_images__image_id__process_retry_post: {
    parameters: {
      path: {
        image_id: number;
      };
    };
    responses: {
      /** @description Successful Response */
      204: {
        content: never;
      };
      /** @description Validation Error */
      422: {
        content: {
          "application/json": components["schemas"]["HTTPValidationError"];
        };
      };
    };
  };
  /** Stop latest process */
  __command_v1_images__image_id__process_stop_delete: {
    parameters: {
      path: {
        image_id: number;
      };
    };
    responses: {
      /** @description Successful Response */
      204: {
        content: never;
      };
      /** @description Validation Error */
      422: {
        content: {
          "application/json": components["schemas"]["HTTPValidationError"];
        };
      };
    };
  };
}
