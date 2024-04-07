import { ProcessFormConfsItem } from "@/features/processes/components/organisms/ProcessFormOrg/ProcessFormConfs/ProcessFormConfsItem";
import { ProcessFormConfsTabs } from "@/features/processes/components/organisms/ProcessFormOrg/ProcessFormConfs/ProcessFormConfsTabs";
import { useProcessFormConfs } from "@/features/processes/components/organisms/ProcessFormOrg/ProcessFormConfs/useProcessFormConfs";
import { ButtonMol } from "@/features/shared/components/molecules/ButtonMol";
import { HorizontalRadioMol } from "@/features/shared/components/molecules/HorizontalRadioMol";
import { InputIntMol } from "@/features/shared/components/molecules/InputIntMol";
import { ToggleSwitchMol } from "@/features/shared/components/molecules/ToggleSwitchMol";
import { components } from "@/services/backend/endpoints";

interface Props {
  card: components["schemas"]["CardMod"];
  onSuccessSubmit?: () => void;
}

export function ProcessFormConfs({ card, onSuccessSubmit }: Props) {
  const {
    configurations,
    setExtension,
    setScalingType,
    setPreserveRatio,
    setScalingBicubicTargetWidth,
    setScalingBicubicTargetHeight,
    setScalingAIScale,
    runProcess,
    isPending,
  } = useProcessFormConfs(card, onSuccessSubmit);

  return (
    <div className="space-y-4">
      <ProcessFormConfsItem label="Source" disabled>
        <div className="flex items-center space-x-1">
          <InputIntMol value={card.source.width} disabled className="w-12" />
          <span>x</span>
          <InputIntMol value={card.source.height} disabled className="w-12" />
          <span>px</span>
        </div>
      </ProcessFormConfsItem>
      <ProcessFormConfsItem label="Extension">
        <HorizontalRadioMol
          possibleValues={["JPEG", "PNG", "WEBP"]}
          value={configurations.extension}
          setValue={setExtension}
          className="w-40"
        />
      </ProcessFormConfsItem>
      <ProcessFormConfsTabs
        tabs={["Bicubic", "AI"]}
        defaultTab={configurations.scalingType}
        onActiveTabChange={setScalingType}
      >
        <div className="space-y-4">
          <ProcessFormConfsItem label="Preserve ratio">
            <ToggleSwitchMol
              checked={configurations.scalingBicubic.preserve_ratio}
              onSwitch={setPreserveRatio}
            />
          </ProcessFormConfsItem>
          <ProcessFormConfsItem label="Target">
            <div className="flex items-center space-x-1">
              <InputIntMol
                value={configurations.scalingBicubic.target.width}
                min={1}
                max={1920}
                onChange={setScalingBicubicTargetWidth}
                className="w-12"
              />
              <span>x</span>
              <InputIntMol
                value={configurations.scalingBicubic.target.height}
                min={1}
                max={1920}
                onChange={setScalingBicubicTargetHeight}
                className="w-12"
              />
              <span>px</span>
            </div>
          </ProcessFormConfsItem>
        </div>
        <div className="space-y-4">
          <ProcessFormConfsItem label="Extension">
            <HorizontalRadioMol
              possibleValues={[2, 3, 4]}
              value={configurations.scalingAI.scale}
              setValue={setScalingAIScale}
              className="w-40"
            />
          </ProcessFormConfsItem>
          <ProcessFormConfsItem label="Target" disabled>
            <div className="flex items-center space-x-1">
              <InputIntMol
                value={card.source.width * configurations.scalingAI.scale}
                disabled
                className="w-12"
              />
              <span>x</span>
              <InputIntMol
                value={card.source.height * configurations.scalingAI.scale}
                disabled
                className="w-12"
              />
              <span>px</span>
            </div>
          </ProcessFormConfsItem>
        </div>
      </ProcessFormConfsTabs>
      <ButtonMol
        label="Let's run!"
        isLoading={isPending}
        onClick={() => runProcess()}
      />
    </div>
  );
}
