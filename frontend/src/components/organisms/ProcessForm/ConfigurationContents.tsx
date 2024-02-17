import { useBackend } from "../../../services/backend";
import { components, paths } from "../../../services/backend/endpoints";
import { Button } from "../../molecules/Button";
import { HorizontalRadio } from "../../molecules/HorizontalRadio";
import { InputInt } from "../../molecules/InputNumber";
import { ToggleSwitch } from "../../molecules/ToggleSwitch";
import { LabeledConfig } from "./LabeledConfig";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

interface Props {
  image: components["schemas"]["ImageODto"];
  latestProcess?: components["schemas"]["ProcessODto"];
  onSuccessSubmit?: () => void;
}

export function ConfigurationContents({
  image,
  latestProcess,
  onSuccessSubmit,
}: Props) {
  const { backend } = useBackend();
  const queryClient = useQueryClient();
  const { mutate: runProcess, isPending } = useMutation({
    mutationFn: () =>
      backend
        .post<
          paths["/images/{id}/process"]["post"]["responses"]["200"]["content"]["application/json"]
        >(`/images/${image.id}/process`, configurations)
        .then((_) => _.data),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: [`/images/${image.id}/process`],
      });
      onSuccessSubmit && onSuccessSubmit();
    },
  });

  const [configurations, setConfigurations] = useState<
    paths["/images/{id}/process"]["post"]["requestBody"]["content"]["application/json"]
  >(
    latestProcess
      ? {
          extension: latestProcess.extension,
          target: latestProcess.target,
          enable_ai: latestProcess.enable_ai,
        }
      : { extension: image.extension, target: image.source, enable_ai: false }
  );
  const [preserveRatio, setPreserveRatio] = useState(true);

  const handleExtensionChange = (extension: "JPEG" | "PNG" | "WEBP") =>
    setConfigurations({ ...configurations, extension });
  const handleTargetWidthChange = (newWidth: number) => {
    let newHeight = preserveRatio
      ? Math.round(image.source.height * (newWidth / image.source.width))
      : configurations.target.height;
    newHeight = Math.min(9999, Math.max(1, newHeight));
    setConfigurations({
      ...configurations,
      target: { width: newWidth, height: newHeight },
    });
  };
  const handleTargetHeightChange = (newHeight: number) => {
    let newWidth = preserveRatio
      ? Math.round(image.source.width * (newHeight / image.source.height))
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
          <InputInt value={image.source.width} disabled className="w-12" />
          <span>x</span>
          <InputInt value={image.source.height} disabled className="w-12" />
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
