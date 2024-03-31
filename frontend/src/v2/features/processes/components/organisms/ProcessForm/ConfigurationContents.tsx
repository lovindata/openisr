import { LabeledConfig } from "@/v2/features/processes/components/organisms/ProcessForm/LabeledConfig";
import { Button } from "@/v2/features/shared/components/molecules/Button";
import { HorizontalRadio } from "@/v2/features/shared/components/molecules/HorizontalRadio";
import { InputInt } from "@/v2/features/shared/components/molecules/InputInt";
import { ToggleSwitch } from "@/v2/features/shared/components/molecules/ToggleSwitch";
import { useBackend } from "@/v2/services/backend";
import { components, paths } from "@/v2/services/backend/endpoints";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

interface Props {
  card: components["schemas"]["CardMod"];
  onSuccessSubmit?: () => void;
}

export function ConfigurationContents({ card, onSuccessSubmit }: Props) {
  const { backend } = useBackend();
  const queryClient = useQueryClient();
  const { mutate: runProcess, isPending } = useMutation({
    mutationFn: () =>
      backend
        .post(`/command/v1/images/${card.image_id}/process/run`, configurations)
        .then(() => {}),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: ["/query/v1/app/cards"],
      });
      onSuccessSubmit && onSuccessSubmit();
    },
  });

  const [configurations, setConfigurations] = useState<
    paths["/command/v1/images/{image_id}/process/run"]["post"]["requestBody"]["content"]["application/json"]
  >({
    extension: card.extension,
    target: card.target ? card.target : card.source,
    enable_ai: card.enable_ai,
  });
  const [preserveRatio, setPreserveRatio] = useState(card.preserve_ratio);

  const handleExtensionChange = (extension: "JPEG" | "PNG" | "WEBP") =>
    setConfigurations({ ...configurations, extension });
  const handleTargetWidthChange = (newWidth: number) => {
    let newHeight = preserveRatio
      ? Math.round(card.source.height * (newWidth / card.source.width))
      : configurations.target.height;
    newHeight = Math.min(9999, Math.max(1, newHeight));
    setConfigurations({
      ...configurations,
      target: { width: newWidth, height: newHeight },
    });
  };
  const handleTargetHeightChange = (newHeight: number) => {
    let newWidth = preserveRatio
      ? Math.round(card.source.width * (newHeight / card.source.height))
      : configurations.target.width;
    newWidth = Math.min(9999, Math.max(1, newWidth));
    setConfigurations({
      ...configurations,
      target: { width: newWidth, height: newHeight },
    });
  };
  const handleEnableAIChange = (value: boolean) =>
    setConfigurations({ ...configurations, enable_ai: value });

  return (
    <div className="space-y-3">
      <LabeledConfig label="Source" disabled>
        <div className="flex items-center space-x-1">
          <InputInt value={card.source.width} disabled className="w-12" />
          <span>x</span>
          <InputInt value={card.source.height} disabled className="w-12" />
          <span>px</span>
        </div>
      </LabeledConfig>
      <LabeledConfig label="Extension">
        <HorizontalRadio
          possibleValues={["JPEG", "PNG", "WEBP"]}
          value={configurations.extension}
          setValue={handleExtensionChange}
          className="w-40"
        />
      </LabeledConfig>
      <LabeledConfig label="Preserve ratio">
        <ToggleSwitch checked={preserveRatio} onSwitch={setPreserveRatio} />
      </LabeledConfig>
      <LabeledConfig label="Target">
        <div className="flex items-center space-x-1">
          <InputInt
            value={configurations.target.width}
            min={1}
            max={9999}
            onChange={handleTargetWidthChange}
            className="w-12"
          />
          <span>x</span>
          <InputInt
            value={configurations.target.height}
            min={1}
            max={9999}
            onChange={handleTargetHeightChange}
            className="w-12"
          />
          <span>px</span>
        </div>
      </LabeledConfig>
      <LabeledConfig label="Enable AI (only on upscale)">
        <ToggleSwitch
          checked={configurations.enable_ai}
          onSwitch={handleEnableAIChange}
        />
      </LabeledConfig>
      <Button
        label="Let's run!"
        isLoading={isPending}
        onClick={() => runProcess()}
      />
    </div>
  );
}
