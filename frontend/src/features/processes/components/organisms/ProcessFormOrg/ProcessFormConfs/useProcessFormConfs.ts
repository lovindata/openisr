import { useBackend } from "@/services/backend";
import { components, paths } from "@/services/backend/endpoints";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

export function useProcessFormConfs(
  card: components["schemas"]["CardMod"],
  onSuccessSubmit?: () => void
) {
  const { backend } = useBackend();
  const queryClient = useQueryClient();
  const { mutate, isPending } = useMutation({
    mutationFn: (
      body: paths["/commands/v1/images/{image_id}/process/run"]["post"]["requestBody"]["content"]["application/json"]
    ) =>
      backend
        .post(`/commands/v1/images/${card.image_id}/process/run`, body)
        .then(() => {}),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: ["/queries/v1/app/cards"],
      });
      onSuccessSubmit && onSuccessSubmit();
    },
  });

  const [configurations, setConfigurations] = useState({
    extension: card.default_extension,
    scalingType: card.default_scaling_type,
    scalingBicubic: card.default_scaling_bicubic,
    scalingAI: card.default_scaling_ai,
  });

  const setExtension = (extension: "JPEG" | "PNG" | "WEBP") =>
    setConfigurations({ ...configurations, extension });
  const setScalingType = (scalingType: "Bicubic" | "AI") => {
    setConfigurations({
      ...configurations,
      scalingType: scalingType,
      scalingBicubic: card.default_scaling_bicubic,
      scalingAI: card.default_scaling_ai,
    });
  };
  const setPreserveRatio = (preserveRatio: boolean) =>
    setConfigurations({
      ...configurations,
      scalingBicubic: {
        ...configurations.scalingBicubic,
        preserve_ratio: preserveRatio,
      },
    });
  const setScalingBicubicTargetWidth = (newWidth: number) => {
    let newHeight = configurations.scalingBicubic.preserve_ratio
      ? Math.round(card.source.height * (newWidth / card.source.width))
      : configurations.scalingBicubic.target.height;
    newHeight = Math.min(1920, Math.max(1, newHeight));
    setConfigurations({
      ...configurations,
      scalingBicubic: {
        ...configurations.scalingBicubic,
        target: {
          width: newWidth,
          height: newHeight,
        },
      },
    });
  };
  const setScalingBicubicTargetHeight = (newHeight: number) => {
    let newWidth = configurations.scalingBicubic.preserve_ratio
      ? Math.round(card.source.width * (newHeight / card.source.height))
      : configurations.scalingBicubic.target.width;
    newWidth = Math.min(1920, Math.max(1, newWidth));
    setConfigurations({
      ...configurations,
      scalingBicubic: {
        ...configurations.scalingBicubic,
        target: {
          width: newWidth,
          height: newHeight,
        },
      },
    });
  };
  const setScalingAIScale = (scale: 2 | 3 | 4) => {
    setConfigurations({
      ...configurations,
      scalingAI: {
        ...configurations.scalingAI,
        scale: scale,
      },
    });
  };

  const runProcess = () => {
    switch (configurations.scalingType) {
      case "Bicubic":
        mutate({
          extension: configurations.extension,
          scaling: {
            type: "Bicubic",
            width: configurations.scalingBicubic.target.width,
            height: configurations.scalingBicubic.target.height,
          },
        });
        break;
      case "AI":
        mutate({
          extension: configurations.extension,
          scaling: {
            type: "AI",
            scale: configurations.scalingAI.scale,
          },
        });
        break;
    }
  };

  return {
    configurations,
    setExtension,
    setScalingType,
    setPreserveRatio,
    setScalingBicubicTargetWidth,
    setScalingBicubicTargetHeight,
    setScalingAIScale,
    runProcess,
    isPending,
  };
}
